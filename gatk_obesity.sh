#!/bin/bash
gatk_dir=$1
input_dir=$2
output_dir=$3
num_threads=$4
export PATH="$PATH:${gatk_dir}/gatk-4.2.0.0/"
export PATH="$PATH:/home/minhnth/samtools-1.14/"
export PATH=/usr/lib/jvm/java-11-openjdk-11.0.4.11-0.el7_6.x86_64/bin/:$PATH
mkdir $output_dir

for file_name in $input_dir/*
do
  # Split string to get name and extent
  name_extent="${file_name#${file_name%/*}/}"

  # Split string to get name
  name="${name_extent:0:-9}"

  # Trimmomatic
  trimmomatic SE \
    $file_name \
    -threads 40 \
    $output_dir/'trimo_'${name_extent} \
    TRAILING:20 \
    MINLEN:20
  # BWA MEM
  bwa mem $gatk_dir/hg38.fasta -t $num_threads \
    -R "@RG\tID:FLOWCELL1\tPU:LANE1\tPL:IONTORRENT\tLB:LIB-OBL-1\tSM:${name}" \
    $output_dir/'trimo_'${name_extent} \
    > $output_dir/'aln_se_'${name}'.sam'
  # Sort bam
  samtools sort $output_dir/'aln_se_'${name}'.sam' \
    -o $output_dir/'sorted_'${name}'.bam'
  # BaseRecalibrator
  gatk BaseRecalibrator -I $output_dir/'sorted_'${name}'.bam' \
    -R $gatk_dir/hg38.fasta \
    --known-sites $gatk_dir/1000G_omni2.5.hg38.vcf.gz \
    --known-sites $gatk_dir/1000G_phase1.snps.vcf.gz \
    --known-sites $gatk_dir/Axiom_Exome_Plus.vcf.gz \
    --known-sites $gatk_dir/dbsnp138.vcf.gz \
    --known-sites $gatk_dir/hapmap_3.3.hg38.vcf.gz \
    --known-sites $gatk_dir/known_indels.vcf.gz \
    --known-sites $gatk_dir/Mills_and_1000G.vcf.gz \
    -O $output_dir/'recal_'${name}'.table'

  gatk ApplyBQSR \
     -R $gatk_dir/hg38.fasta \
     -I $output_dir/'sorted_'${name}'.bam' \
     --bqsr-recal-file $output_dir/'recal_'${name}'.table' \
     -O $output_dir/'recal_'${name}'.bam'

  samtools index $output_dir/'recal_'${name}'.bam'

  gatk --java-options "-Xmx16g -XX:ParallelGCThreads=64" HaplotypeCaller \
    --native-pair-hmm-threads $num_threads --min-base-quality-score 20 \
    -R $gatk_dir/hg38.fasta \
    -I $output_dir/'recal_'${name}'.bam' \
    -O  $output_dir/'haplo_'${name}'.g.vcf.gz' \
    -ERC GVCF -G StandardAnnotation \
    -G AS_StandardAnnotation -G StandardHCAnnotation

  # Create cohort for GenotypeGVCFs
  echo -e ${name}'\t'${output_dir}/haplo_${name}.g.vcf.gz >> $output_dir/cohort.sample_map
done

mkdir ${output_dir}/tmp
gatk --java-options "-Xmx4g" GenomicsDBImport \
  --genomicsdb-workspace-path my_database \
  -L /work/users/minhnth/gatk/resources_broad_hg38_v0_wgs_calling_regions.hg38.interval_list \
  --reader-threads $num_threads \
  --tmp-dir $output_dir/tmp/ \
  --batch-size 50 \
  --sample-name-map $output_dir/cohort.sample_map

gatk --java-options "-Xmx4g" GenotypeGVCFs \
  -R $gatk_dir/hg38.fasta \
  -V gendb://my_database \
  -O $output_dir/cohort.vcf.gz \
  #-new-qual

# VQSR
gatk --java-options "-Xmx3g -Xms3g" VariantFiltration \
  -V $output_dir/cohort.vcf.gz \
  --filter-expression "ExcessHet > 54.69" \
  --filter-name ExcessHet \
  -O $output_dir/cohort_excesshet.vcf.gz

gatk MakeSitesOnlyVcf \
  -I $output_dir/cohort_excesshet.vcf.gz \
  -O $output_dir/cohort_sitesonly.vcf.gz

gatk --java-options "-Xmx24g -Xms24g" VariantRecalibrator \
  -V  $output_dir/cohort_sitesonly.vcf.gz \
  --trust-all-polymorphic -tranche 100.0 -tranche 99.95 \
  -tranche 99.9 -tranche 99.5 -tranche 99.0 -tranche 97.0 \
  -tranche 96.0 -tranche 95.0 -tranche 94.0 -tranche 93.5 \
  -tranche 93.0 -tranche 92.0 -tranche 91.0 -tranche 90.0 \
  -an FS -an ReadPosRankSum -an QD -an SOR -an DP \
  -mode INDEL --max-gaussians 1 \
  -resource:mills,known=false,training=true,truth=true,prior=12 $gatk_dir/Mills_and_1000G.vcf.gz \
  -resource:axiomPoly,known=false,training=true,truth=false,prior=10 $gatk_dir/Axiom_Exome_Plus.vcf.gz \
  -resource:dbsnp,known=true,training=false,truth=false,prior=2 $gatk_dir/dbsnp138.vcf.gz \
  -O $output_dir/cohort_indels.recal \
  --tranches-file $output_dir/cohort_indels.tranches

gatk --java-options "-Xmx3g -Xms3g" VariantRecalibrator \
  -V $output_dir/cohort_sitesonly.vcf.gz \
  --trust-all-polymorphic \
  -tranche 100.0 -tranche 99.95 -tranche 99.9 \
  -tranche 99.8 -tranche 99.6 -tranche 99.5 \
  -tranche 99.4 -tranche 99.3 -tranche 99.0 \
  -tranche 98.0 -tranche 97.0 -tranche 90.0 \
  -an QD -an MQRankSum -an ReadPosRankSum -an FS -an MQ -an SOR -an DP \
  -mode SNP --max-gaussians 3 \
  -resource:hapmap,known=false,training=true,truth=true,prior=15 $gatk_dir/hapmap_3.3.hg38.vcf.gz \
  -resource:omni,known=false,training=true,truth=true,prior=12 $gatk_dir/1000G_omni2.5.hg38.vcf.gz \
  -resource:1000G,known=false,training=true,truth=false,prior=10 $gatk_dir/1000G_phase1.snps.vcf.gz \
  -resource:dbsnp,known=true,training=false,truth=false,prior=7 $gatk_dir/dbsnp138.vcf.gz \
  -O $output_dir/cohort_snps.recal \
  --tranches-file $output_dir/cohort_snps.tranches

gatk --java-options "-Xmx5g -Xms5g" \
  ApplyVQSR \
  -V $output_dir/cohort_excesshet.vcf.gz \
  --recal-file $output_dir/cohort_indels.recal \
  --tranches-file $output_dir/cohort_indels.tranches \
  --truth-sensitivity-filter-level 99.7 \
  --create-output-variant-index true \
  -mode INDEL \
  -O $output_dir/indel.recalibrated.vcf.gz

gatk --java-options "-Xmx5g -Xms5g" \
  ApplyVQSR \
  -V $output_dir/indel.recalibrated.vcf.gz \
  --recal-file $output_dir/cohort_snps.recal \
  --tranches-file $output_dir/cohort_snps.tranches \
  --truth-sensitivity-filter-level 99.7 \
  --create-output-variant-index true \
  -mode SNP \
  -O $output_dir/snp_indels.recalibrated.vcf.gz

# Select variant without tags
gatk SelectVariants \
  -V $output_dir/snp_indels.recalibrated.vcf.gz \
  --exclude-filtered true \
  -O $output_dir/snp_indels.filtered.vcf.gz

# Split SNP
gatk SelectVariants \
  -V $output_dir/snp_indels.filtered.vcf.gz \
  -select-type SNP \
  -O $output_dir/snps.filtered.vcf.gz

# Annotation
# snpEff
mkdir $output_dir/annotation
mkdir $output_dir/annotation/snpEff
java -Xmx16g -jar $gatk_dir/snpEff/SnpSift.jar annotate $gatk_dir/dbsnp138.vcf \
  -dbsnp $output_dir/snps.filtered.vcf.gz > $output_dir/annotation/snpEff/snps.dbSnp.vcf

java -Xmx16g -jar $gatk_dir/snpEff/snpEff.jar \
  -v -s $output_dir/annotation/snpEff/snpEff_snps.html \
  -canon hg38 \
  $output_dir/annotation/snpEff/snps.dbSnp.vcf > $output_dir/annotation/snpEff/snpEff_snps.vcf

# snpSift
java -Xmx16g -jar $gatk_dir/snpEff/SnpSift.jar \
  extractFields -s "," -e "." \
  $output_dir/annotation/snpEff/snpEff_snps.vcf \
  CHROM POS ID REF ALT AF DP "ANN[*].GENE" "ANN[*].GENEID" "ANN[*].EFFECT" "ANN[*].IMPACT" "ANN[*].FEATURE" "ANN[*].FEATUREID" "ANN[*].BIOTYPE" "ANN[*].HGVS_C" "ANN[*].HGVS_P" > $output_dir/annotation/snpEff/snpEff_snps.txt

# Funcotator
mkdir $output_dir/annotation/funcotator
gatk Funcotator -R $gatk_dir/hg38.fasta \
  -V $output_dir/snps.filtered.vcf.gz \
  -O $output_dir/annotation/funcotator/funco_snps \
  --output-file-format MAF \
  --data-sources-path $gatk_dir/funcotator_dataSources.v1.7.20200521g \
  --ref-version hg38
