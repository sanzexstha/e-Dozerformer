#!/bin/bash

#echo "Running ETTh1..."
#bash ETTh1.sh
#
#echo "Running ETTh2..."
#bash ETTh2.sh

#echo "Running ETTm1..."
#bash ETTm1.sh

#echo "Running ETTm2..."
#bash ETTm2.sh

#
#echo "Running Exchange_rate..."
#bash Exchange_Rate.sh
#
echo "Running Weather..."
bash WTH.sh
# After all the loops finish
python -c "import wandb; wandb.init(project='e-Dozerformer'); wandb.alert(title='Training Complete', text='All runs finished!')"
echo "All experiments completed."
