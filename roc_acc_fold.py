import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

from sklearn.model_selection import KFold
import os, glob, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="noonan+normal", type=str)
    # parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-k", "--kfold", help="returns the number of splitting iterations in the cross-validator.", 
                        default=10, type=int)
    parser.add_argument("-n", "--names_considered", help="names for different types considered, separated by commas", 
                        default="normal,noonan", type=str)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    parser.add_argument("-s", "--use_shuffled_kfold", help="whether to use shuffled kfold.", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-a", "--additional_data_dir", help="where to get the additional data", 
                        default="", type=str)
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, inference=True)
    
    names_considered = args.names_considered.strip().split(',')
    fp_tp = {}
    accuracy = {}
    for name in names_considered:
        fp_tp[name] = [[], []] # fpr_list, tpr_list
        accuracy[name] = []
    
    # prepare folders
    raw_dir = 'raw_112'
    verify_type = 'verify'
    if args.tta:
        verify_type += '_tta'
    if args.use_shuffled_kfold:
        verify_type += '_shuffled'
    train_dir = conf.facebank_path/args.dataset_dir/verify_type/'train'
    test_dir = conf.data_path/'facebank'/args.dataset_dir/verify_type/'test'
    conf.facebank_path = train_dir
    for name in names_considered:
        os.makedirs(str(train_dir) + '/' + name, exist_ok=True)
        os.makedirs(str(test_dir) + '/' + name, exist_ok=True)

    # init kfold
    if args.use_shuffled_kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=6)
    else:
        kf = KFold(n_splits=args.kfold, shuffle=False, random_state=None)

    # collect and split raw data
    data_dict = {}
    idx_gen = {}
    for name in names_considered:
        data_dict[name] = np.array(glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir) + 
                                            '/' + name + '*'))
        idx_gen[name] = kf.split(data_dict[name])

    # threshold_array = np.arange(1.5, 1.6, 0.2)
    # for threshold in threshold_array:
    # mkdir for folder containing verification results
    # th = 'th_' + '{:.2f}'.format(threshold).replace('.', '_')
    # verify_dir = conf.data_path/'facebank'/args.dataset_dir/verify_type/th
    # if not verify_dir.is_dir():
    #     verify_dir.mkdir(parents=True)
    threshold = 1.6
    learner.threshold = threshold #+ 1.0
    
    if conf.device.type == 'cpu': # conf.device.type = 'cpu' for CRC01/02 
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
        # learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded for threshold', threshold)
    
    # count for roc-auc
    counts = {}
    for name in names_considered:
        counts[name] = [0, 0, 0] # #false, #true, #false_positive
        
    # for fold_idx, (train_index, test_index) in enumerate(kf.split(data_dict[names_considered[0]])):
    for fold_idx in range(args.kfold):
        train_set = {}
        test_set = {}
        for name in names_considered:
            (train_index, test_index) = next(idx_gen[name])
            train_set[name], test_set[name] = data_dict[name][train_index], data_dict[name][test_index]

        # remove previous data 
        prev = glob.glob(str(train_dir) + '/*/*')
        for p in prev:
            os.remove(p)
        prev = glob.glob(str(test_dir) + '/*/*')
        for p in prev:
            os.remove(p)
        # save trains to conf.facebank_path/args.dataset_dir/'train' and 
        # tests to conf.data_path/'facebank'/args.dataset_dir/'test'
        for name in names_considered:
            for i in range(len(train_set[name])):
                # print(args.dataset_dir)
                # print('divided' in args.dataset_dir)
                if 'divided' in args.dataset_dir:
                    for img in os.listdir(train_set[name][i]):
                        shutil.copy(train_set[name][i] + os.sep + img, 
                                    ('/'.join(train_set[name][i].strip().split('/')[:-2]) + 
                                        '/' + verify_type + '/train/' + name + os.sep + img))
                else:
                    shutil.copy(train_set[name][i], 
                                train_set[name][i].replace(raw_dir, verify_type + '/train/' + name))
            for i in range(len(test_set[name])):
                if 'divided' in args.dataset_dir:
                    for img in os.listdir(test_set[name][i]):
                        shutil.copy(test_set[name][i] + os.sep + img, 
                                    ('/'.join(test_set[name][i].strip().split('/')[:-2]) + 
                                        '/' + verify_type + '/test/' + name + os.sep + img))
                else:
                    shutil.copy(test_set[name][i], 
                                test_set[name][i].replace(raw_dir, verify_type + '/test/' + name))
        
        if args.additional_data_dir:
            fake_dict = {'noonan':'normal', 'normal':'noonan'}
            full_additional_dir = conf.data_path/'facebank'/'noonan+normal'/args.additional_data_dir
            add_data = glob.glob(str(full_additional_dir) + os.sep + '*.png')
            print('additional:', args.additional_data_dir, len(add_data))
            for name in names_considered:
                for img_f in add_data:
                    if name in img_f.strip().split(os.sep)[-1]:
                        print('source:', img_f)
                        print('copy to:', img_f.replace(str(full_additional_dir), 
                                                        str(train_dir) + os.sep + fake_dict[name]))
                        # print('copy to:', img_f.replace(args.additional_data_dir, 
                        #                                 verify_type + '/train/' + name))
                        shutil.copy(img_f, img_f.replace(str(full_additional_dir), 
                                                        str(train_dir) + os.sep + fake_dict[name]))
        
        print(fold_idx)
        print('datasets ready')

        # prepare_facebank
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')

        # # folder for 1 fold
        # verify_fold_dir = verify_dir/str(fold_idx)
        # if not verify_fold_dir.is_dir():
        #     verify_fold_dir.mkdir(parents=True)
        
        for path in test_dir.iterdir():
            if path.is_file():
                continue
            print(path)
            for fil in path.iterdir():
                print(fil)
                orig_name = ''.join([i for i in fil.name.strip().split('.')[0] if not i.isdigit()])
                if orig_name not in names_considered:
                    print("Un-considered name:", fil.name)
                    continue
                frame = cv2.imread(str(fil))
                image = Image.fromarray(frame)
                faces = [image,]
                bboxes, _ = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                # print('bboxes shape:', bboxes.shape)
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice    
                results, score = learner.infer(conf, faces, targets, args.tta)
                for idx,bbox in enumerate(bboxes):
                    pred_name = names[results[idx] + 1]
                    frame = draw_box_name(bbox, pred_name + '_{:.2f}'.format(score[idx]), frame)
                    if pred_name in fil.name:
                        counts[orig_name][1] += 1
                    else:
                        counts[orig_name][0] += 1
                        if pred_name in names_considered:
                            counts[pred_name][2] += 1
                # cv2.imwrite(str(verify_fold_dir/fil.name), frame)
                # print('save image to', str(verify_fold_dir/fil.name))

    print(counts)

    # for name in names_considered:
    #     # positive = counts[name][1] + counts[name][2]
    #     fp_tp[name][0].append(counts[name][1] / 52.) # total condition positive 52
    #     fp_tp[name][1].append(counts[name][2] / (52. * 1.)) # total condition negative 52 x 1
    #     accuracy[name].append(counts[name][1] / (counts[name][0] + counts[name][1]))
        
    # # plots
    # colors = list(mcolors.TABLEAU_COLORS)
    
    # plt.figure()
    # for i in range(len(names_considered)):
    #     name = names_considered[i]
    #     fp = np.array(fp_tp[name][0])
    #     tp = np.array(fp_tp[name][1])
    #     idxs = np.argsort(fp)
    #     if i%2 != 1:
    #         plt.plot(fp[idxs], tp[idxs], label=name+' ROC curve', color=colors[i])#, linewidth=4)
    #     else:
    #         plt.plot(fp[idxs], tp[idxs], label=name+' ROC curve', color=colors[i], linestyle=':')#, linewidth=4)
    
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.title('ROC Threshold:{:.2f}-{:.2f}'.format(threshold_array[0], threshold_array[-1]))
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left')
    # plt.savefig(str(conf.data_path/'facebank'/args.dataset_dir/verify_type) + '/fp_tp.png')
    # plt.close()

    # plt.figure()
    # for i in range(len(names_considered)):
    #     name = names_considered[i]
    #     if i%2 != 1:
    #         plt.plot(threshold_array, accuracy[name], label=name+' accuracy curve',
    #                  color=colors[i], linewidth=4)
    #     else:
    #         plt.plot(threshold_array, accuracy[name], label=name+' accuracy curve',
    #                  color=colors[i], linestyle=':', linewidth=4)
    
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Threshold')
    # plt.ylabel('Accuracy')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left')
    # plt.savefig(str(conf.data_path/'facebank'/args.dataset_dir/verify_type) + '/accuracy.png')
    # plt.close()
