#!/bin/bash

module load anaconda/2020.11
module load cuda/11.1
source activate pytorch_38

export PYTHONUNBUFFERED=1

# shellcheck disable=SC2164
cd ../data
python  load_data.py
