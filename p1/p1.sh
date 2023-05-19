#!/bin/bash

module purge
module load anaconda3/2021.11

# For Part 1a
python p1.py --inv --N=64

# For Part 1b
python p1.py --conj --N=128
python p1.py --conj --N=256
python p1.py --conj --N=512

