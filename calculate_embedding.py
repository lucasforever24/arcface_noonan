import os
import datetime
import argparse
from PIL import Image
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms as trans

from config import get_config
import deep_gestalt.Learner_gestalt as glearner
from Learner_trans_tf import face_learner
from utils import prepare_names, prepare_facebank, label_binarize
from mtcnn import MTCNN

def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def get_score(target, emb):
    diff = target.unsqueeze(-1) - emb.transpose(1, 0).unsqueeze(0)
    dist = torch.sum(torch.pow(diff, 2), dim=1)
    score = torch.mm(l2_norm(target), l2_norm(emb).transpose(1, 0))
    return dist, score


def evaluate_embedding(conf, model, mtcnn, tta = True):
    train_dir = conf.emore_folder / 'imgs'
    test_dir = conf.emore_folder / 'test'
    embeddings = dict()
    names = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    names.sort()
    name_to_idx = {cls_name: i for i, cls_name in enumerate(names)}

    model.eval()
    for n in names:
        embs = []
        scores = []
        distances = []
        path = os.path.join(train_dir, n)
        file_list = os.listdir(path)
        for file in file_list:
            file_path = os.path.join(path, file)
            try:
                img = Image.open(file_path)
            except:
                continue

            if img.size != (112, 112):
                img = mtcnn.align(img)
            with torch.no_grad():
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embs.append((emb + emb_mirror)/2)
                else:
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    embs.append(emb)

        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0, keepdim=True)

        for v in embs:
            distance, score = get_score(embedding, v)
            distances.append(distance)
            scores.append(score)

        distances = torch.cat(distances)
        scores = torch.cat(scores)

        print("For %s " % n)
        print("(distance_mean, distance_std) is (%f, %f)" % (torch.mean(distances).item(), torch.std(distances).item()))
        print("(theta_mean, theta_std) is (%f, %f)" % (torch.mean(scores).item(), torch.std(scores).item()))

        embeddings[n] = embedding

    print("reference embedding finished")
    for i in range(len(names) - 1):
        j = i + 1
        while j < len(names):
            print(names[i], names[j], ":")
            distance, score = get_score(embeddings[names[i]], embeddings[names[j]])
            print("distance is % f, theta is % f" % (distance.item(), score.item()))
            j += 1


    for n in names:
        scores = []
        distances = []
        embedding = embeddings[n]
        path = os.path.join(test_dir, n)
        file_list = os.listdir(path)
        for file in file_list:
            file_path = os.path.join(path, file)
            try:
                img = Image.open(file_path)
            except:
                continue

            if img.size != (112, 112):
                img = mtcnn.align(img)
            with torch.no_grad():
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    emb = (emb + emb_mirror) / 2
                    distance, score = get_score(embedding, emb)
                    distances.append(distance)
                    scores.append(score)

                else:
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    distance, score = get_score(embedding, emb)
                    distances.append(distance)
                    scores.append(score)

        distances = torch.cat(distances)
        scores = torch.cat(scores)

        print("For %s test set " % n)
        print("(distance_mean, distance_std) is (%f, %f)" % (torch.mean(distances).item(), torch.std(distances).item()))
        print("(theta_mean, theta_std) is (%f, %f)" % (torch.mean(scores).item(), torch.std(scores).item()))


def store_embedding(conf, model, mtcnn, tta = True):
    train_dir = conf.emore_folder / 'imgs'
    embeddings = []
    names = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    names.sort()
    labels = []

    model.eval()
    for n in names:
        path = os.path.join(train_dir, n)
        file_list = os.listdir(path)
        for file in file_list:
            file_path = os.path.join(path, file)
            try:
                img = Image.open(file_path)
            except:
                continue

            if img.size != (112, 112):
                img = mtcnn.align(img)
            with torch.no_grad():
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embeddings.append((emb + emb_mirror) / 2)
                else:
                    emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    embeddings.append(emb)
            labels.append(n)

    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = np.asarray(labels)
    np.save("embedding.npy", embeddings)
    np.save("labels.npy", labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for face embedding evaluation')
    parser.add_argument("-ac", "--arch", help="types of model used for encoder", default="mobile", type=str)
    parser.add_argument("-g", "--gpu_id", help="gpu id to use", default="", type=str)
    args = parser.parse_args()
    conf = get_config(True, args)

    save_path = Path('results/gestalt/train/12/8/2020  11:04:39/moodel.pth')
    # save_path = Path('work_space/save')
    conf.emore_folder = conf.data_path / "faces_emore"
    # data_dir = Path('data/faces_emore/imgs')
    tta = False
    gestalt = False

    mtcnn = MTCNN()
    print('mtcnn loaded')

    if gestalt:
        learner = glearner.face_learner(conf=conf, ext='gestalt_eval')

        # learner.load_state(save_path, True, True)
        store_embedding(conf, learner.model, mtcnn, tta)

    else:
        learner = face_learner(conf=conf, ext='eval')

        # learner.load_state(save_path, False, True)
        store_embedding(conf, learner.model, mtcnn, tta)


