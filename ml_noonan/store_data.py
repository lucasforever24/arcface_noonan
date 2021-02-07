import sys
sys.path.append('../')

import os
import cv2
import numpy as np

from ml_noonan.extract_features import face_feature, detect_landmarks


if __name__ == "__main__":
    names_considered = ['noonan', 'normal', 'others']
    total_data = []  # store arrays for all patients
    data_dir = '../data/facebank/detect/raw_112'
    patients_list = os.listdir(data_dir)
    for ppl in patients_list:
        data = []
        if names_considered[0] in ppl:
            label = 0
        elif names_considered[1] in ppl:
            label = 1
        elif names_considered[2] in ppl:
            label = 2
        else:
            continue
        print(ppl)
        patient_folder = os.path.join(data_dir, ppl)

        image_list = os.listdir(patient_folder)
        for img in image_list:
            img_path = os.path.join(patient_folder, img)
            image_data = cv2.imread(img_path)
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            shape = detect_landmarks(image_data)

            ef = face_feature(shape)
            ef.features.append(label)
            ef.get_shape_distance()
            ef.get_texture_features(gray)

            data = np.array(ef.features)
            print(data.shape)

        total_data.append(data)
    total_data = np.array(total_data)
    print(total_data.shape)
    np.save('nnnmot.npy', total_data)
    print('Finished')



