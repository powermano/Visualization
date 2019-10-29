#!/usr/bin/env bash
file=$1
save=$2
prefix=$3
epoch=$4
gpus=$5
batch_size=$6
input_shape=$7
input_format=$8
data_type=$9
expand_ratio=${10}
is_qnn=${11}
now_dir=${12}
save_feature=${13}
save_badcase=${14}

python ${now_dir}/../../../predict_with_feature.py \
    --file ${file} \
    --save ${save} \
    --prefix ${prefix} \
    --epoch ${epoch} \
    --gpus ${gpus} \
    --batch-size ${batch_size} \
    --input-shape ${input_shape} \
    --input-format ${input_format} \
    --data-type ${data_type} \
    --expand-ratio ${expand_ratio} \
    --is-qnn ${is_qnn} \
    --save-feature ${save_feature} \
    --save-badcase ${save_badcase}

