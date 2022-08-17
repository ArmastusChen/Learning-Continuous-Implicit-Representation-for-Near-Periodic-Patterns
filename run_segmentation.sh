#!/bin/bash

input_path=data/segmentation/input
detected_path=data/segmentation/detected


for input_file in $input_path/*
do
    name=$(basename "$input_file")
    echo $input_file

    CUDA_VISIBLE_DEVICES="1" python NPP_proposal/search.py --datadir $input_file --outdir $detected_path
    CUDA_VISIBLE_DEVICES="1" python NPP_segmentation/train.py --datadir "${detected_path}/${name}" --basedir ./results --p_topk 3
done

