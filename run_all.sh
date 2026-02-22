#!/bin/bash

echo "Running ETTh1..."
bash ETTh1.sh

echo "Running ETTh2..."
bash ETTh2.sh

echo "Running ETTm1..."
bash ETTm1.sh

echo "Running ETTm2..."
bash ETTm2.sh


echo "Running ETTm1..."
bash Exchange_Rate.sh

echo "Running ETTm2..."
bash WTH.sh

echo "All experiments completed."
