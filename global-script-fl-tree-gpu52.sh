#!/bin/bash

# Assign input arguments to variables
scripts_folder=$1
loss_strategy=$2
seed=$3

# Your script with variables and specific host
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.05.sh $seed"
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.10.sh $seed"
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.20.sh $seed"
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.33.sh $seed"
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.45.sh $seed"
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.60.sh $seed"
oarsub -p "gpu='YES' and host='nefgpu52.inria.fr'" -l /gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.80.sh $seed"
