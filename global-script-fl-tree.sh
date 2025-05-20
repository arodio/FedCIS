##!/bin/bash
#
## Assign input arguments to variables
#scripts_folder=$1
#loss_strategy=$2
#seed=$3
#
## Your script with variables
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.05.sh $seed" -t besteffort
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.10.sh $seed" -t besteffort
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.20.sh $seed" -t besteffort
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.33.sh $seed" -t besteffort
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.45.sh $seed" -t besteffort
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.60.sh $seed" -t besteffort
#oarsub -p "gpu='YES' and gpucapability>='7.0'" -l /nodes=1/gpunum=1,walltime=4 -t idempotent -S "./$scripts_folder/$loss_strategy-0.80.sh $seed" -t besteffort


#####################################################EXP-1###############################################################
echo EXP-1 Started >> progress.txt
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.10.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.60.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.20.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.10.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.10.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.33.sh  1 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.20.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.05.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.60.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.20.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.80.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.45.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.05.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.45.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.10.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.80.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.80.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.20.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.33.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.05.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.33.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.45.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.33.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.45.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/serving_rate-0.05.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/flops_prop-0.80.sh 5 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/equal_weight-0.60.sh 5 &
bash scripts-fl-3-9-17-data-aug-p0-d_to_n-2x-seed/err_min_bias_opt_mean-0.60.sh 5 &
wait
echo EXP-1 Finished >> progress.txt
wait
echo EXP-3 Started >> progress.txt
#####################################################EXP-3###############################################################
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.10.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.60.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.60.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.20.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.10.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.10.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.33.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.20.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.05.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.20.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.60.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.20.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.80.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.05.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.45.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.45.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.05.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.45.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.10.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.80.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.80.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.20.sh 5 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.33.sh 5 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.05.sh 5 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.33.sh 5 &
wait
echo EXP-4-3 Started >> progress.txt
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.45.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.10.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.33.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.45.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/serving_rate-0.05.sh 4 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.33.sh 5 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/opt_err_min-0.80.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/flops_prop-0.80.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/equal_weight-0.60.sh 2 &
bash scripts-fl-3-9-17-data-aug-p0-eq-seed/err_min_bias_opt_mean-0.60.sh 3 &
bash scripts-fl-3-9-17-data-aug-p0-strong_bias_l3-seed/serving_rate-0.60.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-strong_bias_l3-seed/equal_weight-0.45.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-strong_bias_l3-seed/serving_rate-0.45.sh 0 &
bash scripts-fl-3-9-17-data-aug-p0-strong_bias_l3-seed/serving_rate-0.80.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-strong_bias_l3-seed/equal_weight-0.80.sh 1 &
bash scripts-fl-3-9-17-data-aug-p0-strong_bias_l3-seed/equal_weight-0.60.sh 1 &
bash scripts-fl-3-9-17-data-aug-p40-strong_bias_l3-seed/serving_rate-0.60.sh 2 &
bash scripts-fl-3-9-17-data-aug-p40-strong_bias_l3-seed/serving_rate-0.45.sh 2 &
bash scripts-fl-3-9-17-data-aug-p40-strong_bias_l3-seed/serving_rate-0.80.sh 2 &
bash scripts-fl-9-13-17-data-aug-p40-strong_bias_l3-cifar100-seed/serving_rate-0.60.sh 3 &
bash scripts-fl-9-13-17-data-aug-p40-strong_bias_l3-cifar100-seed/serving_rate-0.45.sh 3 &
bash scripts-fl-9-13-17-data-aug-p40-strong_bias_l3-cifar100-seed/serving_rate-0.80.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-strong_bias_l3-cifar100-seed/serving_rate-0.60.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-strong_bias_l3-cifar100-seed/equal_weight-0.45.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-strong_bias_l3-cifar100-seed/serving_rate-0.45.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-strong_bias_l3-cifar100-seed/serving_rate-0.80.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-strong_bias_l3-cifar100-seed/equal_weight-0.80.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-strong_bias_l3-cifar100-seed/equal_weight-0.60.sh 5 &
wait
echo EXP-4-3 Finished >> progress.txt
echo EXP-5 Started >> progress.txt
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.60.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.20.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.10.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.10.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.33.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.10.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.20.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.05.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.60.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.20.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.80.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.45.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.05.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.45.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.10.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.80.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.80.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.20.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.33.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.05.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.33.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.45.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.33.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.45.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/serving_rate-0.05.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/flops_prop-0.80.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/equal_weight-0.60.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-d_to_n-2x-cifar100-seed/err_min_bias_opt_mean-0.60.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.10.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.60.sh 5 &
wait
echo EXP-5 Finished >> progress.txt
echo EXP-6 Started >> progress.txt
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.60.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.20.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.10.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.10.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.33.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.20.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.05.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.20.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.60.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.20.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.80.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.05.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.45.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.45.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.05.sh 2 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.45.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.10.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.80.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.80.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.20.sh 3 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.33.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.05.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.33.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.45.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.10.sh 4 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.33.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.45.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/serving_rate-0.05.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.33.sh 5 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/opt_err_min-0.80.sh 5 &
echo EXP-6 Finished >> progress.txt
echo EXP-7 Started >> progress.txt
wait
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/flops_prop-0.80.sh 0 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/equal_weight-0.60.sh 1 &
bash scripts-fl-9-13-17-data-aug-p0-eq-cifar100-seed/err_min_bias_opt_mean-0.60.sh 2 &
wait
echo EXP-7 Finished >> progress.txt



