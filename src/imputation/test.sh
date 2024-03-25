#!/bin/bash
ref_genome_dir=$1 # /work/users/minhnth/gatk/hg38.fasta
input_dir=$2 # ../data/obesity
output_dir=$3 # ../data/obesity/imputation_output/
ref_panel=$4
genetic_map=$5
num_threads=$6

# Create imputed list for all chromosomes
inputed_files=${output_dir}'/imputed_list.txt'
chr_names=${output_dir}'/chr_names.txt'

for CHR in {1..23}; do
    if [ -f ${output_dir}/biallelic_imputed_info_chr${CHR}.vcf.gz ]; then
        echo ${output_dir}/biallelic_imputed_info_chr${CHR}.vcf.gz >> $inputed_files
        echo ${CHR} >> ${chr_names}
    else
        echo biallelic_imputed_info_chr${CHR}.vcf.gz 'does not exist.'
    fi
done

# Concat imputed files

bcftools concat -f ${inputed_files} -Oz -o ${output_dir}/biallelic_combination.vcf
rm ${inputed_files}

# Sort SNPs
bcftools sort ${output_dir}/biallelic_combination.vcf -Oz -o ${input_dir}/biallelic_for_training.vcf

# List existing chromosomes
chr=()
while IFS= read -r line
    do
        chr+=($(($line)))
    done < ${chr_names}
rm ${chr_names}

# Generate an allele frequency file for plotting for each chrs
for CHR in ${chr}; do
    # Generate a header for the output file
    echo -e 'CHR\tSNP\tREF\tALT\tAF\tINFO\tAF_GROUP' \
        > ${output_dir}/biallelic_group_chr${CHR}.txt

    # Query the required fields and 
    # add frequency group (1, 2 or 3) as the last column
    bcftools query -f \
        '%CHROM\t%CHROM\_%POS\_%REF\_%ALT\t%REF\t%ALT\t%INFO/AF\t%INFO/INFO\t-\n' \
        ${output_dir}/biallelic_imputed_info_chr${CHR}.vcf.gz | \
    # Here $5 refers to AF values, $7 refers to AF group
    awk -v OFS="\t" \
        '{if ($5>=0.05 && $5<=0.95) $7=1; \
            else if(($5>=0.005 && $5<0.05) || \
            ($5<=0.995 && $5>0.95)) $7=2; else $7=3} \
            { print $1, $2, $3, $4, $5, $6, $7 }' \
        >> ${output_dir}/biallelic_group_chr${CHR}.txt
done

# Copy and save the given 'plot_INFO_and_AF_for_imputed_chrs.R' file 
# and run it with:
for CHR in ${chr}; do
    Rscript --no-save  \
        plot_INFO_and_AF_for_imputed_chrs.R \
        ${output_dir}/biallelic_group_chr${CHR}.txt \
        ${output_dir}/biallelic_chr${CHR} \
        ${output_dir}/1000GP_imputation_all.frq
done

# Combine the plots per chromosome into a single pdf file
convert $(ls ${output_dir}/_chr*.png | sort -V) \
     ${output_dir}.pdf

