#!/bin/bash

#$ -M jliu16@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N lag_styl_gpu         # Specify job name

module load pytorch

DataDir=divided
LagData=LAG_y_fine
TransDepth=3
Model=srm112df_nn

for Op in "train" "test" "train,test"
do
    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -td ${TransDepth} \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_${Model}_${Op}

    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -s -td ${TransDepth} \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_${Model}_${Op}_s

    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -s -rs 888 -td ${TransDepth} \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_${Model}_${Op}_s888

    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -tta -td ${TransDepth} \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_${Model}_${Op}_tta
done

for Op in "test" "train,test"
do
    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -td ${TransDepth} \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_lag_test_${Model}_${Op}

    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -s -td ${TransDepth} \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_lag_test_${Model}_${Op}_s

    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -s -rs 888 -td ${TransDepth} \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_lag_test_${Model}_${Op}_s888

    python fold_cur_trans_tf.py -ds ${DataDir} -g 0 -tta -td ${TransDepth} \
    -a ${LagData} -ta "test" \
    -as ${Model} -ts ${Op} \
    > data/facebank/trans/plt_recs/trans_${DataDir}_lag_test_${Model}_${Op}_tta
done

# LagData=LAG_y_fine

# for DataDir in srm112df srm112df_no0
# do
#     python fold_cur_reverse.py -ds ${DataDir} -g 0 \
#     > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}

#     python fold_cur_reverse.py -ds ${DataDir} -g 0 -s \
#     > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_s

#     python fold_cur_reverse.py -ds ${DataDir} -g 0 -s -rs 888 \
#     > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_s888

#     python fold_cur_reverse.py -ds ${DataDir} -g 0 -tta \
#     > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_tta

#     for Op in "train" "test" "train,test"
#     do
#         python fold_cur_reverse.py -ds ${DataDir} -g 0 \
#         -a ${LagData} -ta ${Op} \
#         > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_lag_${Op}

#         python fold_cur_reverse.py -ds ${DataDir} -g 0 -s \
#         -a ${LagData} -ta ${Op} \
#         > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_lag_${Op}_s

#         python fold_cur_reverse.py -ds ${DataDir} -g 0 -s -rs 888 \
#         -a ${LagData} -ta ${Op} \
#         > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_lag_${Op}_s888

#         python fold_cur_reverse.py -ds ${DataDir} -g 0 -tta \
#         -a ${LagData} -ta ${Op} \
#         > data/facebank/trans/plt_recs/no_trans_split_nOverlap_${DataDir}_lag_${Op}_tta
#     done
# done

# LagData=LAG_y_fine

# for DataDir in srm112df srm112df_no0
# do
#     python cur_reverse.py -ds ${DataDir} -g 0 \
#     > data/facebank/trans/plt_recs/no_trans_nOverlap_${DataDir}

#     python cur_reverse.py -ds ${DataDir} -g 0 -tta \
#     > data/facebank/trans/plt_recs/no_trans_nOverlap_${DataDir}_tta
# done

# DataDir=srm112df
# #srm112d_no0
# for Op in "train" "test"
# do
#     python cur_reverse.py -d ${DataDir} -g 0 \
#     -a ${LagData} -ta ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}

#     python cur_reverse.py -d ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_tta
# done

# Op=train,test

# python cur_reverse.py -d ${DataDir} -g 0 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs6

# python cur_reverse.py -d ${DataDir} -g 0 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs6_tta

# python cur_reverse.py -d ${DataDir} -g 0 -rs 8 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs8

# python cur_reverse.py -d ${DataDir} -g 0 -rs 8 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs8_tta

# python cur_reverse.py -d ${DataDir} -g 0 -rs 888 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs888

# python cur_reverse.py -d ${DataDir} -g 0 -rs 888 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs888_tta

# DataDir=srm112df_no0

# for Op in "train" "test"
# do
#     python cur_reverse.py -d ${DataDir} -g 0 \
#     -a ${LagData} -ta ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}

#     python cur_reverse.py -d ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_tta
# done

# Op=train,test

# python cur_reverse.py -d ${DataDir} -g 0 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs6

# python cur_reverse.py -d ${DataDir} -g 0 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs6_tta

# python cur_reverse.py -d ${DataDir} -g 0 -rs 8 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs8

# python cur_reverse.py -d ${DataDir} -g 0 -rs 8 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs8_tta

# python cur_reverse.py -d ${DataDir} -g 0 -rs 888 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs888

# python cur_reverse.py -d ${DataDir} -g 0 -rs 888 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs888_tta

# for Op in "train" "test"
# do
#     python cur_reverse.py -d ${DataDir} -g 0 \
#     -a ${LagData} -ta ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}

#     python cur_reverse.py -d ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_tta
# done

# Op=train,test

# python cur_reverse.py -d ${DataDir} -g 0 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs6

# python cur_reverse.py -d ${DataDir} -g 0 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs6_tta

# python cur_reverse.py -d ${DataDir} -g 0 -rs 8 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs8

# python cur_reverse.py -d ${DataDir} -g 0 -rs 8 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs8_tta

# python cur_reverse.py -d ${DataDir} -g 0 -rs 888 \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs888

# python cur_reverse.py -d ${DataDir} -g 0 -rs 888 -tta \
# -a ${LagData} -ta ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_${LagData}_${Op}_rs888_tta

# DataDir=divided
# LagData=LAG_y_fine
# Model=srm112d_no0

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}

#     python fold_cur.py -d ${DataDir} -g 0 -s \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s

#     python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_s888

#     python fold_cur.py -d ${DataDir} -g 0 -tta \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_${Model}_${Op}_tta
# done

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}

#     python fold_cur.py -d ${DataDir} -g 0 -s \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_s

#     python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_s888

#     python fold_cur.py -d ${DataDir} -g 0 -tta \
#     -a ${LagData} -ta ${Op} \
#     -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Model}_${Op}_tta
# done


# DataDir=smile
# LagData=LAG_y_fine
# Model=divi_mtcnn_112

# python fold_cur.py -d ${DataDir} -g 0  -a ${LagData} -ta "train" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train_divi_test

# python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ta "train" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train_divi_test_s

# python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 -a ${LagData} -ta "train" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train_divi_test_s888

# python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -ta "train" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train_divi_test_tta

# DataDir=divided
# LagData=LAG_y_fine
# Model=smile_refine_mtcnn_112_divi
# python fold_cur.py -d ${DataDir} -g 0  -a ${LagData} -ta "test" \
# -as ${Model} -ts "train,test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_styl_train,test

# python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ta "test" \
# -as ${Model} -ts "train,test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_styl_train,test_s

# python fold_cur.py -d ${DataDir} -g 0 -s -rs 888 -a ${LagData} -ta "test" \
# -as ${Model} -ts "train,test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_test_styl_train,test_s888

# python fold_cur.py -d ${DataDir} -g 0  -a ${LagData} -ta "train,test" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train,test_styl_test

# python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ta "train,test" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train,test_styl_test_s

# python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -ta "train,test" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train,test_styl_test_tta

# for K in 5 8
# do
#     for SD in 8 10
#     do
#         python fold_cur.py -d ${DataDir} -g 0 -k ${K} -s -rs ${SD} \
#         > data/facebank/trans/plt_recs/no_trans_${DataDir}_k${K}_rs${SD}_s
#     done
# done

# DataDir=ref_noonan
# python fold_cur.py -d ${DataDir} -g 0  -a ${LagData} -ta "train,test" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train,test_styl_test

# python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ta "train,test" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train,test_styl_test_s

# python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -ta "train,test" \
# -as ${Model} -ts "test" \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_train,test_styl_test_tta

# python fold_cur.py -d ${DataDir} -g 0 \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}
# python fold_cur.py -d ${DataDir} -g 0 -s \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_s
# python fold_cur.py -d ${DataDir} -g 0 -tta \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_tta
# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}_tta
# done

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 -a ${LagData} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}_tta
# done

# for Op in "train" "test" "train,test"
# do
#     python fold_cur.py -d ${DataDir} -g 0 -a ${LagData} -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}_tta
# done

# DataDir=divided
# Op=train
# echo ${Op}
# python fold_cur.py -d ${DataDir} -g 0 -a ${LagData} -as ${Model} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}
# python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -as ${Model} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}_s
# python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -as ${Model} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}_tta
# python fold_cur.py -d ${DataDir} -g 0 -a ${LagData} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}
# python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}_s
# python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}_tta
# python fold_cur.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}
# python fold_cur.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}_s
# python fold_cur.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} \
# > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}_tta

# echo ${DataDir}
# for Op in "test" "train,test"
# do
#     echo ${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -a ${LagData} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_${Op}_tta
#     python fold_cur.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_styl_${Op}_tta
# done

# for Op in "test" "train,test"
# do
#     echo ${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -a ${LagData} -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}
#     python fold_cur.py -d ${DataDir} -g 0 -s -a ${LagData} -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -a ${LagData} -as ${Model} -ts ${Op} \
#     > data/facebank/trans/plt_recs/no_trans_${DataDir}_lag_styl_${Op}_tta
# done

# DataDir=distinct
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/${DataDir}

# TransDepth=2
# Model=smile_refine_mtcnn_112_divi
# echo ${TransDepth}
# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python fold_cur.py -d ${DataDir} -g 0 -t ${TransDepth} \
#     > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}
#     python fold_cur.py -d ${DataDir} -g 0 -s -t ${TransDepth} \
#     > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta -t ${TransDepth} \
#     > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_tta
#     for Op in "train" "test" "train,test"
#     do
#         echo ${Op}
#         python fold_cur.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}
#         python fold_cur.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_s
#         python fold_cur.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} -t ${TransDepth} \
#         > data/facebank/trans/plt_recs/trans_${TransDepth}_${DataDir}_${Model}_${Op}_tta
#     done
# done

# Model=smile_refine_mtcnn_112_divi
# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     for Op in "train" "test" "train,test"
#     do
#         echo ${Op}
#         python fold_cur.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} > data/facebank/plt_recs/retrain_${DataDir}_${Model}_${Op}
#         python fold_cur.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} > data/facebank/plt_recs/retrain_${DataDir}_${Model}_${Op}_s
#         python fold_cur.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} > data/facebank/plt_recs/retrain_${DataDir}_${Model}_${Op}_tta
#     done
# done

# python fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/retrain_${DataDir}
# python fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/retrain_${DataDir}_s
# python fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/retrain_${DataDir}_tta

# Model=smile_refine_mtcnn_112_divi
# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     for Op in "train" "test" "train,test"
#     do
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -as ${Model} -ts ${Op} > data/facebank/plt_recs/${DataDir}_${Model}_${Op}_filter
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s -as ${Model} -ts ${Op} > data/facebank/plt_recs/${DataDir}_${Model}_${Op}_filter_s
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta -as ${Model} -ts ${Op} > data/facebank/plt_recs/${DataDir}_${Model}_${Op}_filter_tta
#     done
# done


# DataDir=stylegan
# echo ${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/${DataDir}_s
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/${DataDir}_tta

# DataDir=divided
# echo ${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/${DataDir}_s
# python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/${DataDir}_tta
# for Model in mix_aug_500_pool5_full mix_aug_500_full mix_aug_100_pool5_full
# do
#     echo ${Model}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
# done

# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/${DataDir}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/${DataDir}_tta
#     for Model in mix_aug_500_pool5_full mix_aug_500_full mix_aug_100_pool5_full
#     do
#         echo ${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
#     done
# done

# DataDir=divided
# echo ${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} > data/facebank/plt_recs/${DataDir}
# python roc_acc_fold_cur.py -d ${DataDir} -s > data/facebank/plt_recs/${DataDir}_s
# python roc_acc_fold_cur.py -d ${DataDir} -tta > data/facebank/plt_recs/${DataDir}_tta
# for Model in mix_aug_500_pool5_full_refine mix_aug_500_full_refine mix_aug_100_pool5_full_refine
# do
#     echo ${Model}
#     python roc_acc_fold_cur.py -d ${DataDir} -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#     python roc_acc_fold_cur.py -d ${DataDir} -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
# done

# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/retrain_${DataDir}
#     python fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/retrain_${DataDir}_s
#     python fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/retrain_${DataDir}_tta
#     for Model in mix_aug_500_pool5_full mix_aug_500_full mix_aug_100_pool5_full
#     do
#         echo ${Model}
#         python fold_cur.py -d ${DataDir} -g 0 -a fake_${Model} > data/facebank/plt_recs/retrain_${DataDir}_${Model}
#         python fold_cur.py -d ${DataDir} -g 0 -s -a fake_${Model} > data/facebank/plt_recs/retrain_${DataDir}_${Model}_s
#         python fold_cur.py -d ${DataDir} -g 0 -tta -a fake_${Model} > data/facebank/plt_recs/retrain_${DataDir}_${Model}_tta
#     done
# done

# for DataDir in distinct divided
# do
#     for Model in mix_aug_500_pool5_full_refine2 mix_aug_500_full_refine2 mix_aug_100_pool5_full_refine2
#     do
#         echo ${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#         python roc_acc_fold_cur.py -d ${DataDir} -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
#     done
# done
# python evaluate_model.py

# for DataDir in distinct divided
# do
#     echo ${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 > data/facebank/plt_recs/${DataDir}
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s > data/facebank/plt_recs/${DataDir}_s
#     python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta > data/facebank/plt_recs/${DataDir}_tta
#     for Model in fr_mix_aug_pix2pix_transfer_b6_500_pool5_full fr_mix_aug_pix2pix_transfer_b6_500_full fr_mix_aug_pix2pix_transfer_b6_100_pool5_full fr_mix_aug_pix2pix_transfer_b6_100_full
#     do
#         echo ${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -s -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_s
#         python roc_acc_fold_cur.py -d ${DataDir} -g 0 -tta -a fake_${Model} > data/facebank/plt_recs/${DataDir}_${Model}_tta
#     done
# done

# for Model in fr_mix_aug_pix2pix_transfer_b6_100_full fr_mix_aug_pix2pix_transfer_b6_2000_D2G fr_mix_pix2pix_transfer_b6_2000_D2G
# do
#     python roc_acc_fold.py -d distinct -g 0 -a fake_${Model} > data/facebank/roc_dist_${Model}
#     python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Model} > data/facebank/roc_dist_${Model}_s
#     python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Model} > data/facebank/roc_dist_${Model}_tta
# done

# Epoch=latest

# for Type in 'raw' 'resize' 'raw_resize'
# do
#     for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
#     do
#         python roc_acc_fold.py -d distinct -g 0 -a fake_${Type}_${Epoch}_${Model} \
#         > data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#         tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#         python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Type}_${Epoch}_${Model} \
#         > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#         tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#         python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Type}_${Epoch}_${Model} \
#         > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#         tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#     done
# done


# for Epoch in 20 latest
# do
#     for Type in 'raw' 'resize' 'raw_resize'
#     do
#         for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
#         do
#             echo "${Type}_${Epoch}_${Model}"
#             python roc_acc_fold.py -d distinct -g 0 -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#             tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}
#             echo "${Type}_${Epoch}_${Model}_s"
#             python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#             tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_s
#             echo "${Type}_${Epoch}_${Model}_tta"
#             python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#             tail -n 1 data/facebank/roc_dist_${Type}_${Epoch}_${Model}_tta
#         done
#     done
# done


# for Model in raw_fr_lag_aug_pix2pix_transfer_b6_100_2layer fr_lag_aug_pix2pix_transfer_b6_100_2layer fr_lag_pix2pix_transfer_b6_100_2layer fr_lag_aug_pix2pix_transfer_b6_500_2layer fr_mix_aug_pix2pix_transfer_b6_500_DG fr_mix_pix2pix_transfer_b6_500_DG
# do
#     echo "${Model}"
#     python roc_acc_fold.py -d distinct -g 0 -a fake_${Model} > data/facebank/roc_dist_${Model}
#     tail -n 2 data/facebank/roc_dist_${Model}
#     echo "${Model}_s"
#     python roc_acc_fold.py -d distinct -g 0 -s -a fake_${Model} > data/facebank/roc_dist_${Model}_s
#     tail -n 2 data/facebank/roc_dist_${Model}_s
#     echo "${Model}_tta"
#     python roc_acc_fold.py -d distinct -g 0 -tta -a fake_${Model} > data/facebank/roc_dist_${Model}_tta
#     tail -n 2 data/facebank/roc_dist_${Model}_tta
# done

# python roc_acc_fold.py -g 0 -s -a fake_fr_aug_pix2pix_transfer_b6_100_2layer > data/facebank/roc_aug_100_2layer_s

# for Epoch in 20 latest
# do
#     for Type in 'raw' 'resize' # 'raw_resize'
#     do
#         for Model in 'fr_adult_pix2pix_transfer_b6_25_1layer_fe' 'fr_adult_pix2pix_transfer_b6_25_2layer' 'fr_adult_pix2pix_transfer_b6_25_1layer_pool5_DG'
#         do
#             python roc_acc_fold.py -g 0 -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_${Type}_${Epoch}_${Model}
#             python roc_acc_fold.py -g 0 -s -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_${Type}_${Epoch}_${Model}_s
#             python roc_acc_fold.py -g 0 -tta -a fake_${Type}_${Epoch}_${Model} \
#             > data/facebank/roc_${Type}_${Epoch}_${Model}_tta
#         done
#     done
# done

#10 15
#'raw_resize'
# for Epoch in 20 latest
# do
#     for Type in 'raw' 'resize'
#     do
#         for Model in '1layer_fe' '2layer' '1layer_pool5_DG'
#         do
#             echo ${Type}_${Epoch}_${Model}
#             tail -n 3 roc_${Type}_${Epoch}_fr_adult_pix2pix_transfer_b6_25_${Model}
#             echo ${Type}_${Epoch}_${Model}_s
#             tail -n 3 roc_${Type}_${Epoch}_fr_adult_pix2pix_transfer_b6_25_${Model}_s
#             echo echo ${Type}_${Epoch}_${Model}_tta
#             tail -n 3 roc_${Type}_${Epoch}_fr_adult_pix2pix_transfer_b6_25_${Model}_tta
#         done
#     done
# done

# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -a "fake_raw" > data/facebank/roc_raw_112_s_a_raw
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta -a "fake_raw" > data/facebank/roc_raw_112_tta_a_raw
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta \
#     -a "fake_raw" > data/facebank/roc_raw_112_s_tta_a_raw
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a "fake_raw" > data/facebank/roc_raw_112_a_raw

# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a "fake_resize" > data/facebank/roc_raw_112_a_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -a "fake_resize" > data/facebank/roc_raw_112_s_a_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta -a "fake_resize" > data/facebank/roc_raw_112_tta_a_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta -a "fake_resize" > data/facebank/roc_raw_112_s_tta_a_resize

# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -a "fake_noonan_resize" > data/facebank/roc_raw_112_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s \
#     -a "fake_noonan_resize" > data/facebank/roc_raw_112_s_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta \
#     -a "fake_noonan_resize" > data/facebank/roc_raw_112_tta_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta \
#     -a "fake_noonan_resize" > data/facebank/roc_raw_112_s_tta_a_noonan_resize
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES > data/facebank/roc_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s > data/facebank/roc_s_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta > data/facebank/roc_tta_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta > data/facebank/roc_s_tta_raw_112
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s > data/facebank/roc_s_0
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -tta > data/facebank/roc_tta_0
# python roc_acc_fold.py -g $CUDA_VISIBLE_DEVICES -s -tta > data/facebank/roc_s_tta_0
# python face_verify_noonan_fold_gpu.py -g $CUDA_VISIBLE_DEVICES > data/facebank/default_2_gpu
# python face_verify_noonan_fold_gpu.py -tta -g $CUDA_VISIBLE_DEVICES > data/facebank/tta_1_gpu
# python face_verify_noonan_fold_gpu.py -s -g $CUDA_VISIBLE_DEVICES > data/facebank/shuffle_1_gpu
# python face_verify_noonan_fold_gpu.py -s -tta -g $CUDA_VISIBLE_DEVICES > data/facebank/shuffle_tta_1_gpu
