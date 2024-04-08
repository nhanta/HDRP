#!/bin/bash
input_dir=$1
output_dir=$2
name_path=$3


mkdir ${output_dir}
# Move samples to new directory
while IFS= read -r line
    do
        cp ${input_dir}"$line".WXS.vcf.1 ${output_dir}
    done < ${name_path}

