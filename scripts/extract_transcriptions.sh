#!/bin/sh
#SBATCH --job-name=hubert_feature
#SBATCH --ntasks-per-node=5
#SBATCH --partition [partition]
#SBATCH --account [account]
#SBATCH --qos [qos]
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/hubert_feature_%J.log
#SBATCH --mail-user=mail@mail.com

rank=$1

module load ffmpeg/4.3.2
module load cuda/11.1
module load cudnn/8.1.1/cuda-11.X

cd aubert
source ~/anaconda3/bin/activate p39
python predict_transformers.py --input-dir ../temp-out-librivox/ --output-dir transcriptions --lm-path 3-gram.pruned.1e-7.arpa  --batch-size 4 --device cuda --num-speakers 200 --rank $rank
