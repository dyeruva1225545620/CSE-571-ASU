#!/bin/sh

module purge
module load python/3.7.1

python3 -m pip install pandas
python3 -m pip install numpy
python3 -m pip install tqdm

python3 automate.py $1
