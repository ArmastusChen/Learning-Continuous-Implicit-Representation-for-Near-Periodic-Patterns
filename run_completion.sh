#!/bin/bash

input_path=data/completion/input
detected_path=data/completion/detected


for input_file in $input_path/*
do
    name=$(basename "$input_file")
    echo $input_file
    echo "${detected_path}/${name}"
    CUDA_VISIBLE_DEVICES="0" python NPP_completion/train.py --datadir "${detected_path}/${name}" --basedir ./results --p_topk 3
done
