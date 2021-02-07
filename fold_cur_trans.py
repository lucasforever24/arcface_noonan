import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner_trans_tf import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank, save_label_score, label_binarize

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-ds", "--dataset_dir", help="where to get data", default="noonan", type=str)
    parser.add_argument('-sd','--stored_result_dir',help='where to store data as np arrays',
                        default="results/trans/", type=str)
    parser.add_argument("-k", "--kfold", help="returns the number of splitting iterations in the cross-validator.", 
                        default=10, type=int)
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-n", "--names_considered", help="names for different types considered, separated by commas", 
                        default="normal,noonan,others", type=str)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    parser.add_argument("-s", "--use_shuffled_kfold", help="whether to use shuffled kfold.", action="store_true")
    parser.add_argument("-rs", "--random_seed", help="random seed used for k-fold split.", default=6, type=int)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-a", "--additional_data_dir", help="where to get the additional data", 
                        default="", type=str)
    parser.add_argument("-ta", "--additional_test_or_train", help="use additional data in only train, or test, or both", 
                        default="", type=str)
    parser.add_argument("-as", "--stylegan_data_dir", help="where to get the additional data", 
                        default="", type=str)
    parser.add_argument("-ts", "--stylegan_test_or_train", help="use stylegan data in only train, or test, or both", 
                        default="", type=str)
    parser.add_argument("-tf", "--transfer", help="how many layer(s) used for transfer learning, "
                        "but 0 means retraining the whole network.", default=0, type=int)
    parser.add_argument("-ac", "--arch", help="types of model used for encoder", default="mobile", type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(arg+':', getattr(args, arg))

    emore_dir = 'faces_emore'
    conf = get_config(True, args)
    conf.emore_folder = conf.data_path/emore_dir

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    names_considered = args.names_considered.strip().split(',')

    exp_name = args.dataset_dir[:4]
    if args.additional_data_dir:
        if 'LAG' in args.additional_data_dir:
            exp_name += '_lag'
        elif 'literature' in args.additional_data_dir:
            exp_name += '_ltr'
    if args.kfold != 10:
        exp_name += ('_k' + str(args.kfold))
    if args.epochs != 20:
        exp_name += ('_e' + str(args.epochs))
    if args.transfer != 0 and args.transfer != 1:
        exp_name += ('_td' + str(args.transfer))
    if args.use_shuffled_kfold:
        exp_name += ('_s' + str(args.random_seed))

    print(exp_name)


    # prepare folders
    raw_dir = 'raw_112'
    verify_type = 'trans'
    if args.use_shuffled_kfold:
        verify_type += '_shuffled'
    # train_dir = conf.facebank_path/args.dataset_dir/verify_type/'train'
    train_dir = conf.emore_folder/'imgs'
    test_dir = conf.emore_folder/'test'
    conf.facebank_path = train_dir

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    for name in names_considered:
        os.makedirs(str(train_dir) + '/' + name, exist_ok=True)
        os.makedirs(str(test_dir) + '/' + name, exist_ok=True)

    if args.stylegan_data_dir:
        #e.g. smile_refine_mtcnn_112_divi
        full_stylegan_dir = str(conf.data_path/'facebank'/'stylegan'/args.stylegan_data_dir)
        stylegan_folders = os.listdir(full_stylegan_dir)
    if args.additional_data_dir:
        full_additional_dir = str(conf.data_path/'facebank'/args.additional_data_dir)

    # init kfold
    if args.use_shuffled_kfold:
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.random_seed)
    else:
        kf = KFold(n_splits=args.kfold, shuffle=False, random_state=None)

    # collect and split raw data
    data_dict = {}
    idx_gen = {}
    for name in names_considered:
        tmp_list = glob.glob(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir) + 
                                            '/' + name + '*')
        if 'innm' in args.stylegan_data_dir:
            tmp_list = tmp_list + glob.glob(str(full_stylegan_dir) + '/' + name + '*')
            stylegan_folders = []
        print(str(conf.data_path/'facebank'/args.dataset_dir/raw_dir))
        data_dict[name] = np.array(tmp_list)
        idx_gen[name] = kf.split(data_dict[name])


    if 'literature' in args.additional_data_dir:
        data_dict['ltr'] = np.array(glob.glob(str(full_additional_dir) + '/*'))
        idx_gen['ltr'] = kf.split(data_dict['ltr'])

    score_names = []
    scores = []
    wrong_names = []

    args.stored_result_path = args.stored_result_dir + os.sep + str(datetime.datetime.now())[:19]
    if not os.path.exists(args.stored_result_path):
        os.mkdir(args.stored_result_path)

    # for fold_idx, (train_index, test_index) in enumerate(kf.split(data_dict[names_considered[0]])):
    for fold_idx in range(args.kfold):
        train_set = {}
        test_set = {}
        for name in names_considered:
            (train_index, test_index) = next(idx_gen[name])
            train_set[name], test_set[name] = data_dict[name][train_index], data_dict[name][test_index]

        if 'ltr' in data_dict.keys():
            (train_index, test_index) = next(idx_gen['ltr'])
            train_set['ltr'], test_set['ltr'] = data_dict['ltr'][train_index], data_dict['ltr'][test_index]
            if 'train' in args.additional_test_or_train:
                train_set['noonan'] = np.concatenate((train_set['noonan'], train_set['ltr']))
            if 'test' in args.additional_test_or_train:
                test_set['noonan'] = np.concatenate((test_set['noonan'], test_set['ltr']))

        # remove previous data 
        prev = glob.glob(str(train_dir) + '/*/*')
        for p in prev:
            os.remove(p)
        prev = glob.glob(str(test_dir) + '/*/*')
        for p in prev:
            os.remove(p)
        # save trains to conf.facebank_path/args.dataset_dir/'train' and 
        # tests to conf.data_path/'facebank'/args.dataset_dir/'test'

        # count unbalanced data
        train_count = {}
        test_count = {}

        for name in names_considered:
            train_count[name] = 0
            for i in range(len(train_set[name])):
                img_folder = str(train_set[name][i])
                for img in os.listdir(img_folder):
                    shutil.copy(img_folder + os.sep + str(img),
                                os.path.join(str(train_dir), name, str(img)))
                    train_count[name] += 1
                # addition data from stylegan
                if 'interp' not in data_dict.keys():
                    folder = os.path.basename(train_set[name][i])
                    if args.stylegan_data_dir and ('train' in args.stylegan_test_or_train) and (folder in stylegan_folders):
                        for img in os.listdir(full_stylegan_dir + os.sep + folder):
                            shutil.copy(os.path.join(full_stylegan_dir, folder, str(img)),
                                        os.path.join(str(train_dir), name, str(img)))
                                        # ('/'.join(train_set[name][i].strip().split('/')[:-2]) + 
                                        #     '/' + verify_type + '/train/' + name + os.sep + img))
                            train_count[name] += 1

            # test
            for i in range(len(test_set[name])):
                test_count[name] = 0
                img_folder = str(test_set[name][i])
                for img in os.listdir(img_folder):
                    shutil.copy(img_folder + os.sep + str(img),
                                os.path.join(str(test_dir), name, str(img)))
                    test_count[name] +=  1
                # addition data from stylegan
                if 'interp' not in data_dict.keys():
                    folder = os.path.basename(test_set[name][i])
                    if args.stylegan_data_dir and ('test' in args.stylegan_test_or_train) and (folder in stylegan_folders):
                        # and 
                        # (folder not in ['noonan7','noonan19','noonan23','normal9','normal20','normal23'])):
                        for img in os.listdir(full_stylegan_dir + os.sep + folder):
                            shutil.copy(os.path.join(full_stylegan_dir, folder, str(img)),
                                        os.path.join(str(test_dir), name, str(img)))
                            test_count[name] += 1

            print(train_count, test_count)
        # deal with unbalanced data
        """
        if train_count['normal'] // train_count['noonan'] > 1:
            aug_num = train_count['normal'] // train_count['noonan'] - 1
            for img in os.listdir(os.path.join(str(train_dir), 'noonan')):
                for aug_idx in range(aug_num):
                    aug_img = img[:img.rfind('.')] + '_' + str(aug_idx) + img[img.rfind('.'):]
                    shutil.copy(os.path.join(str(train_dir), 'noonan', img), 
                                os.path.join(str(train_dir), 'noonan', aug_img))
        """


        if 'fake' in args.additional_data_dir:
            fake_dict = {'noonan':'normal', 'normal':'noonan'}
            full_additional_dir = conf.data_path/'facebank'/'noonan+normal'/args.additional_data_dir
            add_data = glob.glob(str(full_additional_dir) + os.sep + '*.png')
            print('additional:', args.additional_data_dir, len(add_data))
            for name in names_considered:
                for img_f in add_data:
                    if name in img_f.strip().split(os.sep)[-1]:
                        # print('source:', img_f)
                        # print('copy to:', img_f.replace(str(full_additional_dir), 
                        #                                 str(train_dir) + os.sep + fake_dict[name]))
                        # print('copy to:', img_f.replace(args.additional_data_dir, 
                        #                                 verify_type + '/train/' + name))
                        shutil.copy(img_f, os.path.join(str(train_dir), fake_dict[name], os.path.basename(img_f)))


        print(fold_idx)
        print('datasets ready')

        conf_train = get_config(True, args)
        conf_train.emore_folder = conf.data_path/emore_dir
        conf_train.stored_result_dir = args.stored_result_path

        learner = face_learner(conf=conf_train, transfer=args.transfer, ext=exp_name+'_'+str(fold_idx))
        # conf, inference=False, transfer=0

        if args.transfer != 0:
            learner.load_state(conf.save_path, False, True)

        print('learner loaded')

        learner.train(conf_train, args.epochs)
        print('learner retrained.')

        learner.save_state()
        print('Model is saved')
        # prepare_facebank
        targets, names, names_idx = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('names_classes:', names)
        noonan_idx = names_idx['noonan']

        print('facebank updated')
        
        for path in test_dir.iterdir():
            if path.is_file():
                continue
            # print(path)
            for fil in path.iterdir():
                # print(fil)
                orig_name = ''.join([i for i in fil.name.strip().split('.')[0].split('_')[0] if not i.isdigit()])
                for name in names_idx.keys():
                    if name in orig_name:
                        score_names.append(names_idx[name])
                """
                if orig_name not in names_considered:
                    print("Un-considered name:", fil.name)
                    continue
                """
                frame = cv2.imread(str(fil))
                image = Image.fromarray(frame)
                faces = [image,]
                distance = learner.binfer(conf, faces, targets, args.tta)
                label = score_names[-1]
                score = np.exp(distance.dot(-1))
                pred = np.argmax(score, 1)
                if pred != label:
                    wrong_names.append(orig_name)
                scores.append(score)

    score_names = np.array(score_names)
    wrong_names = np.array(wrong_names)
    score_np = np.squeeze(np.array(scores))

    n_classes = score_np.shape[1]
    score_names = label_binarize(score_names, classes=range(n_classes))
    score_sum = np.zeros([score_np.shape[0], 1])

    for i in range(n_classes):
        score_sum += score_np[:, i, None]  # keep the dimension
    relative_scores = (score_np / score_sum)
    total_scores = relative_scores.ravel()
    total_names = score_names.ravel()

    name_path = os.path.join(args.stored_result_path, 'wrong_names.npy')
    save_label_score(name_path, wrong_names)
    label_path = os.path.join(args.stored_result_path, 'labels_trans.npy')
    save_label_score(label_path, score_names)
    score_path = os.path.join(args.stored_result_path, 'scores_trans.npy')
    save_label_score(score_path, relative_scores)
    print('saved!')
    
    # Compute ROC curve and ROC area for noonan
    fpr, tpr, _ = roc_curve(total_names, total_scores)  #scores_np[:, noonan_idx]
    roc_auc = auc(fpr, tpr)

    # For PR curve
    precision, recall, _ = precision_recall_curve(total_names, total_scores)
    average_precision = average_precision_score(total_names, total_scores)

    # plots
    plt.figure()
    colors = list(mcolors.TABLEAU_COLORS)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_{}'.format(exp_name))
    plt.legend(loc="lower right")
    plt.savefig(args.stored_result_path + os.sep + '/fp_tp_{}.png'.format(exp_name))
    plt.close()
    # plt.show()

    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score ({}): AP={:0.4f}'.format(exp_name, average_precision))
    plt.savefig(args.stored_result_path + os.sep + '/pr_{}.png'.format(exp_name))
    plt.close()

