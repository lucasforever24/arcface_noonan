# Arcface_for_Noonan_detection

Pytorch1.7.1 codes for arcface

------

## 1. Intro

- This repo is the source code of paper "Automated Facial Recognition for Noonan Syndrome using Deep Convolutional Neural
Network with Additive Angular Margin Loss" (under review)

------

## 2. Pretrained Models & Performance
The pretrained models are from:
https://github.com/TreB1eN/InsightFace_Pytorch

[Mobilefacenet @ BaiduNetDisk](https://pan.baidu.com/s/1hqNNkcAjQOSxUjofboN6qg), [Mobilefacenet @ OneDrive](https://1drv.ms/u/s!AhMqVPD44cDOhkSMHodSH4rhfb5u)

| LFW(%) | CFP-FF(%) | CFP-FP(%) | AgeDB-30(%) | calfw(%) | cplfw(%) | vgg2_fp(%) |
| ------ | --------- | --------- | ----------- | -------- | -------- | ---------- |
| 0.9918 | 0.9891    | 0.8986    | 0.9347      | 0.9402   | 0.866    | 0.9100     |

## 3. How to use

- download codes

  ```
  git clone git@github.com:lucasforever24/arcface_noonan.git
  ```

### 3.1 Data Preparation

#### 3.1.1 Prepare Facebank 

Place the face images your want to detect in the data/facebank/noonan folder, and guarantee it have a structure like following:

```
data/facebank/noonan
        ---> noonan1/
            ---> noonan1_1.jpg
        ---> normal1/
            ---> normal2_1.jpg
        ---> others1/
            ---> others1_1.jpg
           ---> others1_2.jpg
```
Noonan stands for Noonan syndrome patients, normal stands for healthy children, and others refer to children with other syndromes.

#### 3.1.2 download the pretrained model to work_space/model

If more than 1 image appears in one folder, an average embedding will be calculated


### 3.2 run:
- bash run.sh
- or python fold_cur_trans.py -ds {the folder name in facebank}


## 4. References 

- This repo is mainly inspired by https://github.com/TreB1eN/InsightFace_Pytorch

