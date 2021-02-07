from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
import mxnet as mx
from tqdm import tqdm

from itertools import combinations
from mtcnn import MTCNN
import os, glob, dlib, shutil, random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(str(imgs_folder), train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf):
    if conf.data_mode in ['ms1m', 'concat']:
        ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
        print('ms1m loader generated')
    if conf.data_mode in ['vgg', 'concat']:
        vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
        print('vgg loader generated')        
    if conf.data_mode == 'vgg':
        ds = vgg_ds
        class_num = vgg_class_num
    elif conf.data_mode == 'ms1m':
        ds = ms1m_ds
        class_num = ms1m_class_num
    elif conf.data_mode == 'concat':
        for i,(url,label) in enumerate(vgg_ds.imgs):
            vgg_ds.imgs[i] = (url, label + ms1m_class_num)
        ds = ConcatDataset([ms1m_ds,vgg_ds])
        class_num = vgg_class_num + ms1m_class_num
    elif conf.data_mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder/'imgs')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, 
                        num_workers=conf.num_workers, drop_last=True)
    return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def load_noonan_val_pair(path, rootdir, transform, image_size=[112,112]):
    # path: path for original images, str
    # rootdir: path to store images and issame_list, Path
    mtcnn = MTCNN()
    print('mtcnn loaded')

    if not rootdir.exists():
        rootdir.mkdir()
    images = os.listdir(path)
    comb_size = len(images) * (len(images) - 1) / 2
    pairs = combinations(images, 2)
    data = bcolz.fill([int(comb_size * 2), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    issame_list = np.zeros(int(comb_size))
    i = 0
    for pair in pairs:
        img0 = Image.open(os.path.join(path, pair[0]))
        if img0.size != (112, 112):
            img0 = mtcnn.align(img0)
        img1 = Image.open(os.path.join(path, pair[1]))
        if img1.size != (112, 112):
            img1 = mtcnn.align(img1)
        data[2*i, ...] = transform(img0)
        data[2*i + 1, ...] = transform(img1)
        if ('noonan' in pair[0] and 'noonan' in pair[1]) or ('normal' in pair[0] and 'normal' in pair[1]):
            issame_list[i] = 1
        i += 1
        if i % 1000 == 0:
            print('loading noonan', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame

def load_mx_rec(rec_path):
    save_path = rec_path/'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path/'train.idx'), str(rec_path/'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1,max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        img = Image.fromarray(img)
        label_path = save_path/str(label)
        if not label_path.exists():
            label_path.mkdir()
        img.save(label_path/'{}.jpg'.format(idx), quality=95)

### using system call for directory operatios instead of pathlib as before
def img2lmk(img_path, lmk_path, in_place=False, predictor_path='data/lmk_predictor/shape_predictor_68_face_landmarks.dat'):
    # os.makedirs(lmk_path, exist_ok=True)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    print('predictor loaded')
    
    if 'webface' in img_path:
        data_path = os.path.join(img_path, '*', '*.jpg')
    elif 'LAG' in img_path:
        data_path = os.path.join(img_path, '*.png')
    else:
        data_path = os.path.join(img_path, '*.jpg')
    
    for f in glob.glob(data_path):
        lmk_f = f.replace(img_path, lmk_path)
        # print(lmk_f)
        lmk_dir = os.sep.join(lmk_f.split(os.sep)[:-1])
        # print(lmk_dir)
        os.makedirs(lmk_dir, exist_ok=True)

        img = dlib.load_rgb_image(f)
        # Ask the detector to find the bounding boxes of each face. The 0 in the
        # second argument indicates that we should upsample the image 0 time. Usually, 
        # it's set to 1 to make everything bigger and allow us to detect more faces.
        dets = detector(img, 0)

        if len(dets) != 1:
            print("Processing file: {}".format(f))
            print("Number of faces detected: {}".format(len(dets)))
            continue
        rec = [dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()]
        # print(rec)
        # exit(0)
        
        shape = predictor(img, dets[0])
        points = [(p.x, p.y) for p in shape.parts()]

        if in_place:
            lmk_img = Image.open(f)
            plt.clf()
            plt.imshow(lmk_img)
            plt.scatter([p[0] for p in points], [p[1] for p in points], marker='o', color='r', linewidths=1)
            currentAxisA = plt.gca()
            currentAxisA.axes.get_xaxis().set_visible(False)
            currentAxisA.axes.get_yaxis().set_visible(False)
            currentAxisA.spines['left'].set_color('none')
            currentAxisA.spines['bottom'].set_color('none')
            rect = patches.Rectangle((rec[0], rec[1]), (rec[3] - rec[1]), (rec[2] - rec[0]), 
                                     linewidth=5, edgecolor='g', facecolor='none')
            currentAxisA.add_patch(rect)
            plt.savefig(lmk_f, bbox_inches='tight', pad_inches=0.0)
        else:
            lmk_img = np.ones(img.shape) * img.mean()
            lmk_img = Image.fromarray(lmk_img.astype('uint8'))
            lmk_draw = ImageDraw.Draw(lmk_img)
            lmk_draw.rectangle(rec, outline='black') #'black'
            lmk_draw.point(points, fill='white') #'white'
            del lmk_draw
            lmk_img.save(lmk_f)

### move images from which dlib cannot detect a face to a folder
def mv_no_face_img(record, orig_path, no_face_path):
    with open(record) as f:
        for line in f:
            if "Processing" in line:
                img_path = line.strip().split(' ')[-1]
                no_img_path = img_path.replace(orig_path, no_face_path)
                no_img_dir = os.sep.join(no_img_path.split(os.sep)[:-1])
                os.makedirs(no_img_dir, exist_ok=True)
                shutil.move(img_path, no_img_path)

def positive_num(num):
    return num if num > 0 else 0

def cg_lmk(img_path, lmk_path, predictor_path='data/lmk_predictor/shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    print('predictor loaded')
    
    if 'webface' in img_path:
        data_path = os.path.join(img_path, '*', '*.jpg')
    else:
        data_path = os.path.join(img_path, '*.jpg')
    
    img_names = []
    lmks = []

    for f in glob.glob(data_path):
        lmk_f = f.replace(img_path, lmk_path)
        
        lmk_dir = os.sep.join(lmk_f.split(os.sep)[:-1])
        # print(lmk_dir)
        os.makedirs(lmk_dir, exist_ok=True)

        img = dlib.load_rgb_image(f)
        # Ask the detector to find the bounding boxes of each face. The 0 in the
        # second argument indicates that we should upsample the image 0 time. Usually, 
        # it's set to 1 to make everything bigger and allow us to detect more faces.
        dets = detector(img, 0)

        if len(dets) != 1:
            print("Processing file: {}".format(f))
            print("Number of faces detected: {}".format(len(dets)))
            continue
        rec = [dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()]

        shape = predictor(img, dets[0])
        points = np.array([(p.x, p.y) for p in shape.parts()])

        move = np.zeros(points.shape)
        # move 16-68 down
        move[17:, 1] += int(positive_num(points[8, 1] - points[57, 1]) * 0.3)
        # move 37-48 apart and change 37,46
        move[36:42, 0] += -int(positive_num(points[42, 0] - points[39, 0]) * 0.1)
        move[42:48, 0] += int(positive_num(points[42, 0] - points[39, 0]) * 0.1)
        move[36, 1] += 1 
        move[45, 1] += 1
        # move 28-31 shorter (and closer to 32-36)
        # move[27:31, 1] = np.ceil((points[33, 1] - points[30, 1]) * 0.2)
        move[29, 1] += round(positive_num(points[30, 1] - points[29, 1]) * 0.1)
        move[28, 1] += round(positive_num(points[30, 1] - points[28, 1]) * 0.1)
        move[27, 1] += np.ceil(positive_num(points[30, 1] - points[27, 1]) * 0.1)
        # move 32-26 wider
        move[34, 0] += 1
        move[35, 0] += np.ceil(positive_num(points[35, 0] - points[33, 0]) * 0.1)
        move[32, 0] += -1
        move[31, 0] += -np.ceil(positive_num(points[33, 0] - points[31, 0]) * 0.1)
        # move 49,61,65,55 lower
        move[48, 1] += np.ceil(positive_num(points[59, 1] - points[49, 1]) * 0.1)
        move[60, 1] += int(positive_num(points[67, 1] - points[61, 1]) * 0.1)
        move[54, 1] += np.ceil(positive_num(points[55, 1] - points[53, 1]) * 0.1)
        move[64, 1] += int(positive_num(points[65, 1] - points[63, 1]) * 0.1)
        
        if 'noonan' in f.strip().split(os.sep)[-1]:
            move = -move
            # print('reverse move:', f)

        new_points = [(p[0], p[1]) for p in (points + move)]
        

        lmk_img = np.ones(img.shape) * img.mean()
        lmk_img = Image.fromarray(lmk_img.astype('uint8'))
        lmk_draw = ImageDraw.Draw(lmk_img)
        lmk_draw.rectangle(rec, outline='black') #'black'
        lmk_draw.point(new_points, fill='white') #'white'
        del lmk_draw
        lmk_img.save(lmk_f)

        img_names.append(lmk_f.split(os.sep)[-1])
        lmks.append(new_points)

    np.save(lmk_dir + os.sep + 'img_names.npy', np.array(img_names))
    np.save(lmk_dir + os.sep + 'lmks.npy', np.array(lmks))

def get_vague_faces(source_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    fake_images = [item for item in os.listdir(source_path) if 'fake_B' in item]
    mtcnn = MTCNN()
    for img in fake_images:
        bboxes, _ = mtcnn.align_multi(Image.open(source_path + os.sep + img), 1, 30)
        # the last two params: conf.face_limit, conf.min_face_size
        if bboxes.shape[0] != 1:
            print(img + ':', bboxes.shape)
        else:
            shutil.copy(source_path + os.sep + img, save_path + os.sep + img)

def select_webfaces(dir='data/facebank/webface/trainA', num=100):
    random.seed(666)
    images = []
    for root, _, fnames in sorted(os.walk(dir)): 
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    selected = random.sample(images, num)
    for f in selected:
        shutil.copy(f, 'data/facebank/webface+children/trainA/' + f.strip().split('/')[-1])
        shutil.copy(f.replace('trainA', 'trainB'), 
                    'data/facebank/webface+children/trainB/' + f.strip().split('/')[-1])

def get_train_dataset_gan(imgs_folder, target_folder, target_size):
    # if 'divided' in imgs_folder:
    for sub_folder in os.listdir(imgs_folder):
        os.makedirs(target_folder + os.sep + sub_folder, exist_ok=True)
    # else:
    #     os.makedirs(target_folder, exist_ok=True)

    imgs=[]
    for root, _, fnames in sorted(os.walk(imgs_folder)): 
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                imgs.append((path, fname))
    mtcnn = MTCNN()
    for img in imgs:
        bboxes, _ = mtcnn.align_multi(Image.open(img[0]), 1, 30)
        # the last two params: conf.face_limit, conf.min_face_size
        if bboxes.shape[0] > 0:
            new_img = mtcnn.align(Image.open(img[0]), target_size)
            # if 'divided' in imgs_folder:
            new_img.save(target_folder + os.sep + '/'.join(img[0].strip().split(os.sep)[-2:]))
            # else:
            #     new_img.save(target_folder + os.sep + img[1])

def get_lag_y_data(lag_data, lag_y_data):
    os.makedirs(lag_y_data, exist_ok=True)
    names = os.listdir(lag_data)
    for name in names:
        y_folder = lag_data + os.sep + name + os.sep + 'y'
        if os.path.isdir(y_folder):
            imgs = os.listdir(y_folder)
            for i in range(len(imgs)):
                shutil.copy(y_folder + os.sep + imgs[i], 
                            lag_y_data + os.sep + name + '_' + str(i) + '.' + imgs[i].strip().split('.')[-1])

def img2lmk_np(img_path, lmk_path, predictor_path='data/lmk_predictor/shape_predictor_68_face_landmarks.dat'):
    os.makedirs(lmk_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    print('predictor loaded')
    
    if 'webface' in img_path:
        data_path = os.path.join(img_path, '*', '*.jpg')
    elif 'LAG' in img_path:
        data_path = os.path.join(img_path, '*.png')
    else:
        data_path = os.path.join(img_path, '*.jpg')
    
    img_names = []
    lmks = []

    for f in glob.glob(data_path):
        img = dlib.load_rgb_image(f)
        # Ask the detector to find the bounding boxes of each face. The 0 in the
        # second argument indicates that we should upsample the image 0 time. Usually, 
        # it's set to 1 to make everything bigger and allow us to detect more faces.
        dets = detector(img, 0)

        if len(dets) != 1:
            print("Processing file: {}".format(f))
            print("Number of faces detected: {}".format(len(dets)))
            continue
        
        rec = [dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()]
        shape = predictor(img, dets[0])
        points = np.array([(p.x, p.y) for p in shape.parts()])

        img_names.append(f.strip().split(os.sep)[-1])
        lmks.append(points)

    np.save(lmk_path + os.sep + 'img_names.npy', np.array(img_names))
    np.save(lmk_path + os.sep + 'lmks.npy', np.array(lmks))

def merge_plt_old(exp_name="divi_retrain", rec_path='data/facebank/plt_recs'):
    # tmp_idx = [48, 49, 50] #+ [33, 34, 35]
    work_idx = [i + 66 + 12 for i in range(12)] #+ tmp_idx # + [15, 16, 17]
    # 0,1,2 basic dist
    # 0 - 14 all dist
    # 15,16,17 basic divi
    # 15 - 29 all divi
    # 30 - 38 dist, refine
    # 39 - 47 divi, refine
    # 48 - 56 dist, refine2
    # 57 - 65 divi, refine2
    # retrain + 66 (missed one fake dataset)
    # 0,1,2 basic dist
    # 0 - 11 all dist
    # 12,13,14 basic divi
    # 12 - 23 all divi

    names = np.load(rec_path + os.sep + 'names.npy')
    labels = np.load(rec_path + os.sep + 'labels.npy', allow_pickle=True)
    scores = np.load(rec_path + os.sep + 'scores.npy', allow_pickle=True)

    colors = list(mcolors.TABLEAU_COLORS)
    color_size = len(colors)
    linestyles = ['--', '-.', ':']
    line_size = len(linestyles)
    lw = 2
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    for (i, idx) in enumerate(work_idx): #names.shape[0]
        fpr, tpr, _ = roc_curve(labels[idx], scores[idx])#scores_np[:, noonan_idx]
        roc_auc = auc(fpr, tpr)
        if idx in [0, 1, 2, 15, 16, 17, 66, 67, 68, 78, 79, 80]:
            plt.plot(fpr, tpr, color=colors[i%color_size], lw=lw, linestyle='-', 
                    label='{} (area = {:0.4f})'.format(names[idx], roc_auc))
        else:
            plt.plot(fpr, tpr, color=colors[i%color_size], lw=lw, linestyle=linestyles[i//color_size], 
                    label='{} (area = {:0.4f})'.format(names[idx], roc_auc))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves ({})'.format(exp_name))
    plt.legend(loc="lower right")
    # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)#, ncol=2)#loc='upper right', 
    # plt.subplots_adjust(right=0.5, top=0.6)
    plt.savefig(rec_path + os.sep + '/fp_tp_{}.png'.format(exp_name))

    plt.figure()
    for (i, idx) in enumerate(work_idx): #names.shape[0]
        precision, recall, _ = precision_recall_curve(labels[idx], scores[idx])
        average_precision = average_precision_score(labels[idx], scores[idx])
        if idx in [0, 1, 2, 15, 16, 17, 66, 67, 68, 78, 79, 80]:
            plt.step(recall, precision, where='post', color=colors[i%color_size], lw=lw, 
                linestyle='-', label='{} (AP={:0.4f})'.format(names[idx], average_precision))
        else:
            plt.step(recall, precision, where='post', color=colors[i%color_size], lw=lw, 
                linestyle=linestyles[i//color_size], label='{} (AP={:0.4f})'.format(names[idx], average_precision))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PR Curves ({})'.format(exp_name))
    plt.legend(loc="lower left")
    plt.savefig(rec_path + os.sep + '/pr_{}.png'.format(exp_name))

def get_sg_faces(source_path, save_path):
    # st: stylegan
    # os.makedirs(save_path, exist_ok=True)
    fake_images = [item for item in os.listdir(source_path)]
    mtcnn = MTCNN()
    for img in fake_images:
        bboxes, _ = mtcnn.align_multi(Image.open(source_path + os.sep + img), 1, 30)
        # the last two params: conf.face_limit, conf.min_face_size
        if bboxes.shape[0] != 1:
            print(img + ':', bboxes.shape)
        else:
            new_img = mtcnn.align(Image.open(source_path + os.sep + img), (112,112))
            subfolder = os.path.basename(img).split('_')[0]
            os.makedirs(save_path + os.sep + subfolder, exist_ok=True)
            new_img.save(os.path.join(save_path, subfolder, img))
            # shutil.copy(source_path + os.sep + img, os.path.join(save_path, subfolder, img))

def map_divi(divi_path):
    map_divi2sg = {}
    with open('data/facebank/sg_divi_name_map.csv') as f:
        for line in f:
            if 'no' in line:
                [sg_name, divi_name] = line.strip().split(',')
                if 'normal9' in divi_name:
                    map_divi2sg[divi_name] = 'normal51'
                else:
                    map_divi2sg[divi_name] = sg_name
    print('map size:', len(map_divi2sg))

    os.makedirs(divi_path + '_tmp', exist_ok=True)
    for divi in map_divi2sg.keys():
        shutil.copytree(divi_path + os.sep + divi, divi_path + '_tmp' + os.sep + map_divi2sg[divi])

def merge_plt(work_idx, exp_name, rec_path='data/facebank/plt_recs'):
    # _dist, _divi, _styl
    # 3 orig, 9 data_aug, 3 stylegan_aug
    # 0,1,2 dist
    # 3 - 11 dist, data_aug
    # 12 - 14 dist_styl
    # + 15 for divi
    # 30, 31, 32 stylegan only
    # 33 - 38 dist_styl
    # 39 - 44 divi_styl
    # 45 - 53 dist_styl_filter
    # 54 - 62 divi_styl_filter
    # retrain, whole network
    # 63,64,65 dist
    # 66,67,68 divi
    # 69 dist_re_smile_refine_mtcnn_112_divi_test (messed up)
    # 70 dist_re_smile_refine_mtcnn_112_divi_test_tta (messed up)
    # 71 - 79 divi_styl
    # data_name = [] #'dist'
    names = np.load(rec_path + os.sep + 'names.npy')[work_idx]
    labels = np.load(rec_path + os.sep + 'labels.npy', allow_pickle=True)[work_idx]
    scores = np.load(rec_path + os.sep + 'scores.npy', allow_pickle=True)[work_idx]
    # for dn in data_name:
    #     if 'styl' in dn:
    #         names.append(np.load(rec_path + os.sep + 'names_{}.npy'.format(dn)))
    #         labels.append(np.load(rec_path + os.sep + 'labels_{}.npy'.format(dn)))
    #         scores.append(np.load(rec_path + os.sep + 'scores_{}.npy'.format(dn)))
    #     else:
    #         names.append(np.load(rec_path + os.sep + 'names_{}.npy'.format(dn))[:3])
    #         labels.append(np.load(rec_path + os.sep + 'labels_{}.npy'.format(dn))[:3, :])
    #         scores.append(np.load(rec_path + os.sep + 'scores_{}.npy'.format(dn))[:3, :])
    #         names.append(np.load(rec_path + os.sep + 'names_{}.npy'.format(dn))[-3:])
    #         labels.append(np.load(rec_path + os.sep + 'labels_{}.npy'.format(dn))[-3:, :])
    #         scores.append(np.load(rec_path + os.sep + 'scores_{}.npy'.format(dn))[-3:, :])

    # total_len = sum([item.shape[0] for item in names])
    # work_idx = [i for i in range(total_len)] #+ tmp_idx # + [15, 16, 17]

    colors = list(mcolors.TABLEAU_COLORS)
    color_size = len(colors)
    linestyles = ['--', '-.', ':']
    line_size = len(linestyles)
    lw = 2
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    i = 0
    # for j in range(len(names)):
    for k in range(names.shape[0]):
        fpr, tpr, _ = roc_curve(labels[k], scores[k])#scores_np[:, noonan_idx]
        roc_auc = auc(fpr, tpr)
        if names[k] in ['distinct', 'distinct_s', 'distinct_tta', 'divided', 'divided_s', 'divided_tta',]:
            #k in [0, 1, 2]:
            plt.plot(fpr, tpr, color=colors[i%color_size], lw=lw, linestyle='-', 
                    label='{} (area = {:0.2f})'.format(names[k], roc_auc))
        else:
            plt.plot(fpr, tpr, color=colors[i%color_size], lw=lw, linestyle=linestyles[i//color_size], 
                    label='{} (area = {:0.2f})'.format(names[k], roc_auc))
        i += 1
    
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves ({})'.format(exp_name))
    plt.legend(loc="lower right")
    # plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)#, ncol=2)#loc='upper right', 
    # plt.subplots_adjust(right=0.5, top=0.6)
    plt.savefig(rec_path + os.sep + '/fp_tp_{}.png'.format(exp_name))

    plt.figure()

    i = 0
    # for j in range(len(names)):
    for k in range(names.shape[0]):
        precision, recall, _ = precision_recall_curve(labels[k], scores[k])
        average_precision = average_precision_score(labels[k], scores[k])
        if names[k] in ['distinct', 'distinct_s', 'distinct_tta', 'divided', 'divided_s', 'divided_tta',]:
            # if k in [0, 1, 2]:
            plt.step(recall, precision, where='post', color=colors[i%color_size], lw=lw, 
                linestyle='-', label='{} (AP={:0.4f})'.format(names[k], average_precision))
        else:
            plt.step(recall, precision, where='post', color=colors[i%color_size], lw=lw, 
                linestyle=linestyles[i//color_size], label='{} (AP={:0.4f})'.format(names[k], average_precision))
        i += 1
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('PR Curves ({})'.format(exp_name))
    plt.legend(loc="lower left")
    plt.savefig(rec_path + os.sep + '/pr_{}.png'.format(exp_name))
# class train_dataset(Dataset):
#     def __init__(self, imgs_bcolz, label_bcolz, h_flip=True):
#         self.imgs = bcolz.carray(rootdir = imgs_bcolz)
#         self.labels = bcolz.carray(rootdir = label_bcolz)
#         self.h_flip = h_flip
#         self.length = len(self.imgs) - 1
#         if h_flip:
#             self.transform = trans.Compose([
#                 trans.ToPILImage(),
#                 trans.RandomHorizontalFlip(),
#                 trans.ToTensor(),
#                 trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])
#         self.class_num = self.labels[-1] + 1
        
#     def __len__(self):
#         return self.length
    
#     def __getitem__(self, index):
#         img = torch.tensor(self.imgs[index+1], dtype=torch.float)
#         label = torch.tensor(self.labels[index+1], dtype=torch.long)
#         if self.h_flip:
#             img = de_preprocess(img)
#             img = self.transform(img)
#         return img, label