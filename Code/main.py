# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:24:10 2021

@author: flopes
"""

from NN import *
from Game import * 
from GameController import * 
from GeneticFunctions import * 

import numpy as np
import time
from datetime import datetime
import os
import shutil
import pickle
import json
import sys

base_dir = "C:\\Users\flopes\\Desktop\\NovaIMS\\CIFO\\Project\\Code"
FOLDER = f"{base_dir}\\{datetime.now().strftime('%Y%m%d%H%M%S')}"

if len(sys.argv) > 1:
    configfile = f"{base_dir}\\{sys.argv[1]}"
else:
    configfile = f"{base_dir}\\config.json"
    
def create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    
create_dir(FOLDER)

# =============================================================================
# Possible Strategies
# Selection:
#     Tournament
#     FPS
#     Rank
# Crossover
#     WholeArithmetic
#     Uniform
#     SinglePoint
#     TwoPoint
# Mutation
#     RandomValue
#     NormalDistributionMutation
#     Inversion
# =============================================================================
    

with open(configfile, "r", encoding="utf-8") as outfile:
    config = json.load(outfile)

with open(f'{FOLDER}\\config.json', "w", encoding="utf-8") as outfile:
    json.dump(config, outfile, indent=2)

##### ------------------------ #####
print(config)
# Read Config and define Variables
nChilds = int(config["Number Of Childs"])
nGenerations = int(config["Number Of Generations"])
CrossoverProbability = float(config["Crossover Probability"])
MutationProbability = float(config["Mutation Probability"])

# Define Genetic Functions
if config["Selection Strategy"] == "Tournament":
    fSelection = TournamentSelection
elif config["Selection Strategy"] == "FPS":
    fSelection = FPSSelection
elif config["Selection Strategy"] == "Rank":
    fSelection = RankSelection
else:
    print("Not configured. Defaulting to Tournament.")
    fSelection = TournamentSelection
    
if config["Mutation Strategy"] == "NormalDistributionMutation":
    fMutation = Mutate_NormalDistributionMutation
elif config["Mutation Strategy"] == "RandomValue":
    fMutation = Mutate_RandomValue
elif config["Mutation Strategy"] == "Inversion":
    fMutation = Mutate_Inversion
else:
    print("Not configured. Defaulting to NormalDistributionMutation")
    fMutation = Mutate_NormalDistributionMutation
    
if config["Crossover Strategy"] == "WholeArithmetic":
    fCrossover = Crossover_WholeArithmetic
elif config["Crossover Strategy"] == "Uniform":
    fCrossover = Crossover_Uniform
elif config["Crossover Strategy"] == "SinglePoint":
    fCrossover = Crossover_SinglePoint
elif config["Crossover Strategy"] == "TwoPoint":
    fCrossover = Crossover_TwoPoint
else:
    print("Not configured. Defaulting to WholeArithmetic.")
    fCrossover = Crossover_WholeArithmetic
    
if config["Maximize Fitness"]:
    fitMaximize = True
else:
    fitMaximize = False
##### ------------------------ #####

# Defining the population size.
PopulationSize = (nChilds,nWeights)
# Creating Population, random at first
np.random.seed(42)
Population = np.random.choice(np.arange(-1,1,step=0.01),size=PopulationSize,replace=True)

durations = []
GameResults = []
FitnessResults = []
lastGenMaxScore = -9e99
lastGenBestGame = None
for genID in range(nGenerations):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gen {genID}")
    GenResult = []
    maxScore = 0
    start = time.time()
    for childID in range(nChilds):
        thisgame = SnakeGame(genID,childID)
        result = playSnake(thisgame,Population[childID])
        GenResult.append( result )
        if result[0] >= maxScore:
            maxScore = result[0]
            if genID == nGenerations-1:
                lastGenBestGame = thisgame
    
    fitness = ComputeFitness(GenResult,fitMaximize)
    GameResults.append(GenResult)
    FitnessResults.append(fitness)
    
    
    Population = fSelection(   fitness,
                               Population,
                               fCrossover,
                               fMutation,
                               CrossoverProbability,
                               MutationProbability,
                               fitMaximize
                           )
    
    dur = (time.time() - start) / 60
    durations.append(dur)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Max Score: {maxScore} Max Fitness: {max(fitness)}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Duration: {dur} minutes.")

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FINISHED! Duration: {sum(durations)} minutes.")
print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Colecting Results.")

GenVector = np.arange(nGenerations)
maxScore = []
avgScore = []
minScore = []

for genId, results in enumerate(GameResults):
    Scores = [ x[0] for x in results]
    maxScore.append(max(Scores))
    avgScore.append(sum(Scores)/len(Scores))
    minScore.append(min(Scores))
    
maxFit = []
avgFit = []
minFit = []

for genId, results in enumerate(FitnessResults):
    maxFit.append(np.max(results))
    avgFit.append(np.mean(results))
    minFit.append(np.min(results))

pypy = False # Place True if pypy is being used as the python engine.
if pypy:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving pkl of best child.")
    with open(f'{FOLDER}\\Game.pkl', 'wb') as f:
        pickle.dump(lastGenBestGame, f)
        
    data = {
        "GenVector" : GenVector,
        "maxScore" : maxScore,
        "avgScore" : avgScore,
        "minScore" : minScore,
        "maxFit" : maxFit,
        "avgFit" : avgFit,
        "minFit" : minFit
        }
    
    with open(f'{FOLDER}\\Data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
else:
    import matplotlib.pyplot as plt
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving pkl of best child.")
    with open(f'{FOLDER}\\Game.pkl', 'wb') as f:
        pickle.dump(lastGenBestGame, f)
        
    data = {
        "GenVector" : GenVector,
        "maxScore" : maxScore,
        "avgScore" : avgScore,
        "minScore" : minScore,
        "maxFit" : maxFit,
        "avgFit" : avgFit,
        "minFit" : minFit
        }
    
    with open(f'{FOLDER}\\Data.pkl', 'wb') as f:
        pickle.dump(data, f)
        
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving mp4 of best child.")
    lastGenBestGame.TimeWalker(FOLDER,"BEST",True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating Score Figure.")
    # Figure 1
    plt.figure(figsize=(20,10))
    plt.plot(GenVector, maxScore, "go--", linewidth = 3,
            markersize = 10, label = "Maximum Score")
     
    plt.plot(GenVector, avgScore, "yo--", linewidth = 3,
            markersize = 10, label = "Average Score")
    
    plt.plot(GenVector, minScore, "bo--", linewidth = 3,
            markersize = 10, label = "Minimum Score")
     
    plt.title("Score through the Generations", fontsize=15)
    plt.xlabel("Generation",fontsize=10)
    plt.ylabel("Score",fontsize=10)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig(f'{FOLDER}\\Score.png', format='png')
    
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating Fitness Figure.")
    # Figure 2
    plt.figure(figsize=(20,10))
    plt.plot(GenVector, maxFit, "go--", linewidth = 3,
            markersize = 10, label = "Maximum Fitness")
     
    plt.plot(GenVector, avgFit, "yo--", linewidth = 3,
            markersize = 10, label = "Average Fitness")
    
    plt.plot(GenVector, minFit, "bo--", linewidth = 3,
            markersize = 10, label = "Minimum Fitness")
     
    plt.title("Fitness through the Generations", fontsize=15)
    plt.xlabel("Generation",fontsize=10)
    plt.ylabel("Fitness",fontsize=10)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.savefig(f'{FOLDER}\\Fitness.png', format='png')
