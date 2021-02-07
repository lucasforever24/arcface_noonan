import os
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import os
import shutil
import cv2
# from extract_regions import detect_landmarks
# from deep_gestalt.dataset.face_utils import save_four_roi


def rename_folder(orig_dir, name='othersnew'):
    folder_list = os.listdir(orig_dir)
    for ppl in folder_list:
        if name not in ppl:
            patient_dir = os.path.join(orig_dir, ppl)
            image_list = os.listdir(patient_dir)
            for i, f in enumerate(image_list):

                new_name = name + ppl + '_' + str(i) + '.jpg'
                os.rename(os.path.join(patient_dir, f), os.path.join(patient_dir, new_name))
                print(new_name)
            new_patient = name + ppl
            os.rename(patient_dir, os.path.join(orig_dir, new_patient))
            print(new_patient)


def align_images(base_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    mtcnn = MTCNN()
    patients_list = os.listdir(base_dir)
    for ppl in patients_list:
        patient_folder = os.path.join(base_dir, ppl)
        new_folder = os.path.join(output_dir, ppl)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
            print("Creating new folder:", new_folder)

            img_list = os.listdir(patient_folder)
            for img in img_list:
                image = Image.open(os.path.join(patient_folder, img)).convert('RGB')
                if image.size != (112, 112):
                    new_image = mtcnn.align(image)
                    # new_image = image.resize((112, 112))
                    new_image.save(os.path.join(new_folder, img))
                    print(new_image.size)
                    print('Saving', img)


def divided_to_distinct(source_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    patients_list = os.listdir(source_dir)
    for ppl in patients_list:
        patient_folder = os.path.join(source_dir, ppl)
        new_folder = os.path.join(dst_dir, ppl)

        if not os.path.exists(new_folder):
            print(ppl)
            os.mkdir(new_folder)
            image_list = os.listdir(patient_folder)
            shutil.copy(os.path.join(patient_folder, image_list[0]), os.path.join(new_folder, image_list[0]))
            print('Copy %s to new folder %s' % (image_list[0], new_folder))


def divided_to_detect(source_dir,  dst_dir):
    # check whether the face in the image is detetable, if yes, move it to the /detect folder in distinct way
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    patients_list = os.listdir(source_dir)
    for ppl in patients_list:
        patient_folder = os.path.join(source_dir, ppl)
        new_folder = os.path.join(dst_dir, ppl)

        if not os.path.exists(new_folder):
            print(ppl)
            image_list = os.listdir(patient_folder)
            for img in image_list:
                orig_path = os.path.join(patient_folder, img)
                image_data = cv2.imread(orig_path)
                b = detect_landmarks(image_data)
                if b:
                    os.mkdir(new_folder)
                    shutil.copy(orig_path, os.path.join(new_folder, img))
                    print('Successful detection of ', img)
                    break
    print('Finished!')


def prepare_smile_test(target_dir, base_dir, output_dir, mode='0.0'):
    # only focus on images that are already in the faces_emore/test directory, simulate test with augmentation
    folder_list = os.listdir(target_dir)
    for set in folder_list:
        output_folder = os.path.join(output_dir, set)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        ppl_list = os.listdir(os.path.join(target_dir, set))
        for ppl in ppl_list:
            if '_' in ppl:
                ppl = ppl.split("_")[0]
            else:
                ppl = ppl.split('.')[0]
            print(ppl)
            base_ppl = os.path.join(base_dir, ppl)
            if os.path.exists(base_ppl):
                img_list = os.listdir(base_ppl)
                for img in img_list:
                    if mode in img:
                        shutil.copy(os.path.join(base_ppl, img), os.path.join(output_folder, img))
                        print('img')


def prepare_smile_all(target_dir, base_dir, output_dir, mode='0.0'):
    # aim at all smiling image
    folder_list = os.listdir(target_dir)
    for f in folder_list:
        output_folder = os.path.join(output_dir, f)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        ppl_list = os.listdir(base_dir)
        for ppl in ppl_list:
            if f in ppl:
                print(ppl)
                base_ppl = os.path.join(base_dir, ppl)
                img_list = os.listdir(base_ppl)
                for img in img_list:
                    if mode in img:
                        shutil.copy(os.path.join(base_ppl, img), os.path.join(output_folder, img))
                        print(img)


def convert_bgr_to_rgb(base_dir):
    folder_list = os.listdir(base_dir)
    for f in folder_list:
        print(f)
        folder = os.path.join(base_dir, f)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            img = cv2.imread(img_path)
            img_array = np.array(img)
            new_img = Image.fromarray(img_array)
            new_img.save(img_path)


if __name__ == "__main__":

    src_dir = '../stylegan-encoder/data/detect/smile/'
    orig_dir = "data/facebank/smile/orig"
    output_dir = "data/facebank/smile/test"
    target_dir = 'data/faces_emore/test'
    base_dir = "data/facebank/webface/trainB"

    convert_bgr_to_rgb(base_dir)
    # align_images(src_dir, orig_dir)
    #  prepare_smile_all(target_dir, base_dir, output_dir, mode='0.0')

    # rename_folder(source_dir_2, name='noonanhospital')

    # align_images(source_dir_1, output_dir_2)
    # align_images(source_dir_2, output_dir_2)

    # divided_to_distinct(output_dir_1, "data/facebank/distinct/literature")
    # divided_to_distinct(output_dir_2, "data/facebank/distinct/raw_112")

    # move directory
    # src_dir = "data/facebank/distinct/new_added/new_noonan/literature"

    # divided_to_detect(output_dir_2, "data/facebank/detect/raw_112")
    # save_four_roi("data/facebank/detect/literature")


    """
    ######### check whether the folder is empty ###########
    dst_dir = "data/facebank/distinct/raw_112"
    patients_list = os.listdir(dst_dir)
    for ppl in patients_list:
        patient_folder = os.path.join(dst_dir, ppl)

        image_list = os.listdir(patient_folder)
        if len(image_list) == 0:
            print(ppl)
    """
            








