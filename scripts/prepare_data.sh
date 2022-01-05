#!/bin/sh
#SBATCH --job-name=aubert_data
#SBATCH --ntasks-per-node 8
#SBATCH --partition [partition]
#SBATCH --account [your account]
#SBATCH --qos [your_qos]
#SBATCH --mem 80G
#SBATCH --time 24:00:00
#SBATCH --output logs/aubert_prepare_pretraining_%J.log
#SBATCH --mail-user mail@mail.com

rank=$1

module load ffmpeg/4.3.2
module load cuda/11.1
module load cudnn/8.1.1/cuda-11.X

cd aubert
source ~/anaconda3/bin/activate p39
python aubert/prepare_data.py --alignments data/aligned_annotations/ --output-dir bin/aubert-data/ --max-ngram-size 4 --num-workers 4 --chunk-size 512 --max-ngrams-per-speaker 512 --rank $rank --world_size 12
