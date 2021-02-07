#!/bin/bash

#$ -M xhu7@nd.edu      # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 16            # Specify parallel environment and legal core size
#$ -q gpu
#$ -l gpu_card=1
#$ -N gestalt_all_trans_gpu         # Specify job name

module load conda
source activate fnn

# DataDir=divided
DataDir=detect
LagData=LAG_y_fine
TransDepth=3
AddDir=distinct/new_added/hospital
NamesConsidered="others,noonan"

# Model=inn05_112
# inn05_112
#  inn05_112 inn05_sf_112

Model=mobile

# python fold_cur_trans.py -ds ${DataDir} -g 0 -td ${TransDepth} \
# -a distinct/new_added/hospital -ta ${Op} \
# > data/facebank/trans/plt_recs/trans_${DataDir}_hospital_${Op}


python3 fold_cur_gestalt.py -g 0 -e 70 -tf 1 -ac ${Model} \
> plt_recs/retrain/trans_${DataDir}_${Model}

python3 fold_cur_gestalt.py -g 0 -e 70 -tf 2 -ac ${Model} \
> plt_recs/retrain/trans_${DataDir}_${Model}

python3 fold_cur_gestalt.py -g 0 -e 70 -tf 3 -ac ${Model} \
> plt_recs/retrain/trans_${DataDir}_${Model}
