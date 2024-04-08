#!/bin/bash
input_dir=$1
output_dir=$2
gatk_dir=$3
ref_dir=$4
ref_name=$5
$result_dir=$6

# Create output directory
mkdir $output_dir
mkdir $result_dir

# Sort vcf files in genome reference dict
for file_name in $input_dir/*
do
    name=${file_name#${file_name%/*}/}
    picard SortVcf \
    I=$file_name \
    O=$output_dir/${name:0:-3}sorted.vcf \
    SEQUENCE_DICTIONARY=$ref_dir/$ref_name.dict \

    echo -e $output_dir/${name:0:-3}sorted.vcf >> $output_dir/vcfs.list
done

# Merge vcf files
java -jar $gatk_dir/GenomeAnalysisTK.jar \
-T CombineVariants -R $ref_dir/$ref_name.fna \
--variant $output_dir/vcfs.list \
-o $output_dir/combined_genotype.vcf \
-genotypeMergeOptions UNIQUIFY

# Query genotype
bcftools query -f '%CHROM %POS %ID %REF %ALT [ %GT]\n' \
$result_dir/combined_genotype.vcf \
-o $output_dir/geno_snps

rm $output_dir/vcfs.list
