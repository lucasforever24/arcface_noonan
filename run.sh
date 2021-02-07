#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N balanced_trans_gpu         # Specify job name

module load conda
source activate fnn

# DataDir=divided
DataDir=distinct
LagData=LAG_y_fine
TransDepth=3
AddDir=distinct/new_added/hospital
NamesConsidered="others, noonan"

# Model=inn05_112
# inn05_112
#  inn05_112 inn05_sf_112

Model=srm112df_nn

# python fold_cur_trans.py -ds ${DataDir} -g 0 -td ${TransDepth} \
# -a distinct/new_added/hospital -ta ${Op} \
# > data/facebank/trans/plt_recs/trans_${DataDir}_hospital_${Op}


for Op in "train" "test" "train,test"
    do
        python fold_cur.py -ds ${DataDir} -g 0 -ta ${Op} -e 100\
        -a ${DataDir}/literature \
        > plt_recs/trans/trans_${DataDir}_${Model}_${Op}
    done
