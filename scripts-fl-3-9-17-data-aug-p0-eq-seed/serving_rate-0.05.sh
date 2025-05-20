#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate federatedML
gpuIndex=$1
gpus=$(nvidia-smi --list-gpus | wc -l)
weight_decay=$(echo "5 * 10^(-4)" | bc -l)
seed=${1:-42}

for eta in 0.1; do
  python main.py --resnet-layers_per_classifier 3-9-17 \
                 --epochs 100 \
                 --resnet-layers-widths 3-2-2-2 \
                 --data-root data/ \
                 --simulator-task train \
                 --data cifar10 \
                 --arch resnet \
                 --specs-dataset-split specs/configuration-cifar10-federated-3-layer-tree-eq.json \
                 --specs-topology specs/3-layer-tree-5-15-80.json \
                 --batch-size 128 \
                 --lr $eta \
                 --weight-decay $weight_decay \
                 --lr-type cosine \
                 --gpu $gpuIndex \
                 --save fl-3-9-17-data-aug-p0-eq-$seed/serving_rate-5/c-0-lr-$eta \
                 --seed $seed \
                 --sampled-probability-per-layer 0.0-0.0 \
                 --lambda-tilde-strategy 0 \
  & ((gpuIndex = (gpuIndex + 1) % gpus))
done


FAIL=0
for job in $(jobs -p); do
  echo $job
  wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ]; then
  echo "Successful"
else
  echo "Unsuccessful! ($FAIL)"
fi
