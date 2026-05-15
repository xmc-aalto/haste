#!/bin/bash
#SBATCH --job-name=AT1p3M_embed_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --output=out/AT1p3M_embed_test.log

echo -e "RUNNING ON:\n"
nvidia-smi
echo -e "\n\n"

module load mamba
source activate sparse_experiments_pytorch
module load triton/2024.1-gcc
module load gcc

export TORCH_LIB_DIR=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH

python src/embed_extraction2.py

#python src/main.py dataset.model.private_feature.head_dropout=0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95 --multirun 


