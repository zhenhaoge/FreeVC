#!/bin/bash
#
# Zhenhao Ge, 2024-06-16

WORK_DIR=$HOME/code/repo/free-vc

# set text file (fid|source-wav|target-wav)
recording_id='MARCHE_AssessmentTacticalEnvironment'
voice='dmytro'
stress='dictionary'
txtpath=$WORK_DIR/txtfiles/${recording_id}_${voice}-${stress}.txt

# set model
# ptfile=$WORK_DIR/checkpoints/freevc.pth # freevc
ptfile=$WORK_DIR/checkpoints/24kHz/freevc-24.pth # freevc-24
[ ! -f $ptfile ] && echo "model $ptfile does not exist!" && exit 1

# set output dir
ptbase=$(basename  ${ptfile%.*})
outdir=$WORK_DIR/outputs/${recording_id}/${ptbase}_${voice}-${stress}
echo "output dir: $outdir"

device=2

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

