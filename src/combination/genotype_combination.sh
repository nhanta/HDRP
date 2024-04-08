#!/bin/bash
input_dir=$1
output_dir=$2
gatk_dir=$3
ref_dir=$4
ref_name=$5


# Create output directory
mkdir $output_dir

# Sort vcf files in genome reference dict
for file_name in $input_dir/*
do
    echo -e $file_name >> $output_dir/vcfs.list
done

# Merge vcf files
java -jar $gatk_dir/GenomeAnalysisTK.jar \
-T CombineVariants -R $ref_dir/$ref_name.fasta \
--variant $output_dir/vcfs.list \
-o $output_dir/combined_genotype.vcf \
-genotypeMergeOptions UNIQUIFY

rm $output_dir/vcfs.list
