#!/bin/bash
file_name=$1
path=$2
vim ${file_name} '+set ff=unix' +wq
while read -r line; do
# reading each line
#echo $line
wget -P ${path} ${line}
done < $file_name