#!/bin/bash
input=$1
output=$2
for folder_name in $input/*
do
  folder_extent_name="${folder_name#${folder_name%/*}/}"
  for file_name in $folder_name/*
  do
    mkdir "${output}"
    mkdir "${output}/${folder_extent_name}"
    name_extent="${file_name#${file_name%/*}/}"
    gunzip -c $file_name | sed -E "s/^((@|\+)SRR[^.]+\.[^.]+)\.(1|2)/\1/" | bgzip -c > ${output}/${folder_extent_name}/${name_extent}
  done
done
