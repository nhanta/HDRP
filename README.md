# An Integrated Method for Disease Risk Prediction Using Next-Generation Sequencing Data
This study presents a preprocessing methodology tailored for next-generation sequencing (NGS) data, integrating advanced computational tools. Leveraging the inherent advantages of NGS technology, this methodology ensures the acquisition of high-quality data essential for model training. Consequently, machine learning algorithms and neural networks are deployed to accurately predict disease risk and identify significant genetic variants.
## Tool Installations
### Installing GATK 4.2.0.0
```
wget https://github.com/broadinstitute/gatk/releases/download/4.2.0.0/gatk-4.2.0.0.zip
unzip gatk-4.2.0.0.zip -d /path to your directory/
export PATH=$PATH:/path to your directory/
```
### Installing Samtools
```
wget https://sourceforge.net/projects/samtools/files/samtools/
tar xvjf samtools-1.1.tar.bz2
cd samtools-1.1
make
export PATH=$PATH:/path to your directory/
```
### Installing Other Tools
- `conda install -c conda-forge r-base=4.1.3`
- `conda install samtools bcftools`
- `conda install -c bioconda plink`
- `conda install -c bioconda sra-tools`
- `conda install -c bioconda fastqc`
- `conda install -c bioconda trimmomatic`
- `conda install -c bioconda bwa`
- `conda install -c bioconda tabix`
- `conda install bioconda::picard`
- `conda install vcftools`
- `pip install bed-reader`
- `pip install captum`
- `pip install torch`
## Data Preparation 
Single-end read data includes [139 obesity samples](https://www.ncbi.nlm.nih.gov/Traces/study/?acc=SRP139885&o=acc_s%3Aa) from Ion Torrent. We run [bash script files](src/gatk/) to download data, and extract the data to fastq.
- Download pileup data by sratools: `prefetch --option-file SRR_Acc_List.txt`
- Decompress sra to fastq: `sh sra_to_fastq.sh`
- Fix fastq files: `sh fix_fastq.sh`
## Raw data preprocessing
Implement GATK for **variant calling**:

`sh gatk_obesity.sh [gatk directory] [input directory] [output directory] [number of threads]`

Folders can be arranged as follows:
```
GATK
|-- Work            
|   |-- GATK directory                   Include GRCh38 and related data.
|   `-- GATK 4.2.0.0                     GATK version 4.2.0.0.                 
|   `-- snpEff/Funcotator                To annotate variants.
|-- Input directory                      Include 139 samples, type of fastq.
|-- Output directory                     To generate result files.
```
## Target data preprocessing
The reference panel can be obtained from the [Hg38 reference panel](https://cgl.gi.ucsc.edu/data/giraffe/construction/). For guidance on preprocessing the reference panel, please consult the [Imputation beagle tutorial](https://github.com/adrianodemarino/Imputation_beagle_tutorial).
```
bash imputation\target_imputation.sh \
[hg38 reference genome] \
../../data/obesity \
../../data/obesity/imputation_output/  \
[reference panel directory] \
[genetic map directory] \
numthreads
```

## Prediction

### Data preparing
Execute [obs_preparation.ipynb](src/obs_preparation.ipynb) to preprocess the training and testing data.
### Feature selection
Utilize the following commands to train models RFE and SFE:
```
python training_rfe_sfe.py \
../data/obesity \
../results/obesity
```

Select significant features for the neural network using:
```
python training_fs_nn.py \
../data/obesity \
../results/obesity
```
For other methods, select important features with:
```
python fs_rfe_sfe.py \
../data/obesity \
../results/obesity
```
### Obesity risk prediction
Execute the following command for obesity risk prediction:
```
python prediction.py \
../data/obesity \
../results/obesity
```
### Evaluate performance
Evaluate the performance using:
```
python evaluation.py \
../results/obesity \
../results/obesity
```