#!/bin/bash
echo "Convert sra to fastq"
for filename in sra/*
do
  echo $filename
  fastq-dump --outdir fastq --gzip --skip-technical  --readids --dumpbase --split-3 --clip $filename
done
