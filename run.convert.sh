#!/bin/bash
#
# Zhenhao Ge, 2024-06-16

PRESENT_DIR=${PWD}
WORK_DIR=${1:-/home/users/zge/code/repo/free-vc}

# set currernt dir to the work dir (for importing modules for FreeVC)
cd $WORK_DIR

# set text file (fid|source-wav|target-wav)
recording_id=${2:-'MARCHE_AssessmentTacticalEnvironment'}
voice=${3:-'dmytro'}
stress=${4:-'dictionary'}
txtpath=$WORK_DIR/txtfiles/${recording_id}_${voice}-${stress}.txt

# set GPU device id
device=${5:-1}

# print arguments
echo "current dir: $WORK_DIR"
echo "recording id: ${recording_id}"
echo "voice: ${voice}"
echo "stress: ${stress}"
echo "txt path: ${txtpath}"
echo "device: ${device}"

# set model
# ptfile=$WORK_DIR/checkpoints/freevc.pth # freevc
ptfile=$WORK_DIR/checkpoints/24kHz/freevc-24.pth # freevc-24
[ ! -f $ptfile ] && echo "model $ptfile does not exist!" && exit 1
echo "model file: ${ptfile}"

# set output dir
ptbase=$(basename  ${ptfile%.*})
outdir=$WORK_DIR/outputs/${recording_id}/${ptbase}_${voice}-${stress}
echo "output dir: $outdir"

## run 1: use model freevc.pth

if [ $ptbase = "freevc" ]; then

    # set config
    hpfile=$WORK_DIR/configs/freevc.json
    
    CUDA_VISIBLE_DEVICES=$device python $WORK_DIR/convert.py \
        --hpfile $hpfile \
        --ptfile $ptfile \
        --txtpath $txtpath \
        --outdir $outdir

elif [ $ptbase = "freevc-24" ]; then

    # set config
    hpfile=$WORK_DIR/configs/freevc-24.json

    CUDA_VISIBLE_DEVICES=$device python $WORK_DIR/convert_24.py \
        --hpfile $hpfile \
        --ptfile $ptfile \
        --txtpath $txtpath \
        --outdir $outdir

else
    echo "model $ptfile does not exist!"
fi

# set current dir back to the original
cd $PRESENT_DIR