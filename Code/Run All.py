# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:39:15 2021

@author: flopes
"""
from runpy import run_path
import sys

base_dir = "C:\\Users\flopes\\Desktop\\NovaIMS\\CIFO\\Project\\Code"

main = f"{base_dir}\\main.py"

Selections = ["Tournament","FPS","Rank"]
Crossovers = ["WholeArithmetic","Uniform","SinglePoint","TwoPoint"]
Mutations = ["RandomValue","NormalDistributionMutation","Inversion"]

sys.argv.append("")

run = 1
maxruns = len(Selections) * len(Crossovers) * len(Mutations) 
for Selection in Selections:
    for Crossover in Crossovers:
        for Mutation in Mutations:
            print(f"Doing {run}/{maxruns} {Selection}-{Crossover}-{Mutation}")
            sys.argv[1] = f"configs\\{Selection}_{Crossover}_{Mutation}.json"
            run_path(main)