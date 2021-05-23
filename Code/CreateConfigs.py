# -*- coding: utf-8 -*-
"""
 This code creates all the possible configuration files to test the algorithm.
"""

import json 

base_dir = "C:\\Users\flopes\\Desktop\\NovaIMS\\CIFO\\Project\\Code"

Selections = ["Tournament","FPS","Rank"]
Crossovers = ["WholeArithmetic","Uniform","SinglePoint","TwoPoint"]
Mutations = ["RandomValue","NormalDistributionMutation","Inversion"]

for Selection in Selections:
    for Crossover in Crossovers:
        for Mutation in Mutations:
            config = {
                      "Number Of Childs": 100,
                      "Number Of Generations": 50,
                      "Maximize Fitness": True,
                      "Crossover Strategy": f"{Crossover}",
                      "Crossover Probability": 0.8,
                      "Mutation Strategy": f"{Mutation}",
                      "Mutation Probability": 0.3,
                      "Selection Strategy": f"{Selection}"
                    }
            with open(f'{base_dir}\\configs\\{Selection}_{Crossover}_{Mutation}.json', "w", encoding="utf-8") as outfile:
                json.dump(config, outfile, indent=2)