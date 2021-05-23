# -*- coding: utf-8 -*-

import copy
import numpy as np

def Mutate_NormalDistributionMutation(weights):
    """
    Implements the Normal Distribution Mutation Operator.
    
    With a 3% probability, each gene will have see its value added to a sample
    taken from a normal distribution with mean 0 and standard deviation 0.1
    """
    MutationProbability = 0.03
    
    newWeights = copy.deepcopy(weights)
    for i,weight in enumerate(newWeights):
        if np.random.random_sample() < MutationProbability:
            newWeights[i] = newWeights[i] + np.random.normal(loc=0.0, scale=0.1)
            
    return newWeights

def Mutate_RandomValue(weights):
    """
    Implements the Random Value Mutation Operator.
    
    With a 3% probability, each gene will have see its value replaced by a sample
    taken from a uniform distribution with range [-1,1]
    """
    MutationProbability = 0.03
    
    newWeights = copy.deepcopy(weights)
    for i,weight in enumerate(newWeights):
        if np.random.random_sample() < MutationProbability:
            newWeights[i] = np.random.uniform(-1,1)
            
    return newWeights

def Mutate_Inversion(weights):
    """
    Implements the Inversion Mutation Operator.
    
    With a 3% probability, each gene will have see its value + the subsequent 4
    reversed in order.
    """
    MutationProbability = 0.03
    FlipSize = 4
    
    newWeights = copy.deepcopy(weights)
    for i,weight in enumerate(newWeights):
        if i <= newWeights.size - FlipSize:
            if np.random.random_sample() < MutationProbability:
                newWeights[i:i+FlipSize] = np.flip(newWeights[i:i+FlipSize])
                
    return newWeights
            
def Crossover_WholeArithmetic(weights1,weights2):
    """
    Implements the Whole Arithmetic Crossover Operator.
    
    Each new gene will be the average of both parents gene.
    """
    alfa = 0.5
    newWeights = alfa*weights1 + (1-alfa) * weights2
    
    return newWeights

def Crossover_Uniform(weights1,weights2):
    """
    Implements the Uniform Crossover Operator.
    
    Each gene will be selected from Parent 1 or 2 with a 50% probability.
    """
    newWeights = copy.deepcopy(weights1)
    for i,weight in enumerate(newWeights):
        if np.random.random_sample() < 0.5:
            newWeights[i] = weights2[i]
    return newWeights

def Crossover_SinglePoint(weights1,weights2):
    """
    Implements the Single Point Crossover Operator.
    
    Replaces the tails of both parents with each other in 1 random crossover point.
    """
    # Choose point which is not an extreme of the array
    CrossPoint = np.random.choice(np.arange(1,weights1.size-1))
    
    if np.random.random_sample() < 0.5:
        newWeights = np.concatenate([copy.deepcopy(weights1)[0:CrossPoint],
                                     copy.deepcopy(weights2)[CrossPoint:]])
    else:
        newWeights = np.concatenate([copy.deepcopy(weights2)[0:CrossPoint],
                                     copy.deepcopy(weights1)[CrossPoint:]])
        
    return newWeights

def Crossover_TwoPoint(weights1,weights2):
    """
    Implements the Two Point Crossover Operator.
    
    Replaces the tails of both parents with each other in 2 random crossover point.
    """
    CrossPoints = np.sort(np.random.choice(np.arange(1,weights1.size-1), size=2, replace = False))
    while abs(CrossPoints[0]-CrossPoints[1]) <= 1: # Make sure that the 2 points are not next to each other
        CrossPoints = np.sort(np.random.choice(np.arange(1,weights1.size-1), size=2, replace = False))
    
    
    if np.random.random_sample() < 0.5:
        newWeights = np.concatenate([copy.deepcopy(weights1)[0:CrossPoints[0]],
                                     copy.deepcopy(weights2)[CrossPoints[0]:CrossPoints[1]],
                                     copy.deepcopy(weights1)[CrossPoints[1]:]
                                     ])
        
    else:
        newWeights = np.concatenate([copy.deepcopy(weights2)[0:CrossPoints[0]],
                                     copy.deepcopy(weights1)[CrossPoints[0]:CrossPoints[1]],
                                     copy.deepcopy(weights2)[CrossPoints[1]:]
                                     ])
        
        
    return newWeights

def ComputeFitness(results, Maximize):
    """
    Based on the results obtained, computes the fitness of each child.
    """
    scoreMultiplier = 100000
    movementsMultiplier = 0
    snakelengthMultiplier = 0
    penaltiesMultiplier = -1000000
    rewardsMultiplier = 1000
    
    return np.sum(np.array(results) * np.array([scoreMultiplier,
                                         movementsMultiplier,
                                         snakelengthMultiplier,
                                         penaltiesMultiplier,
                                         rewardsMultiplier]
                                        )
                   , axis=1
                   )

def TournamentSelection(fitness,pop,funcCrossover,funcMutation,CrossProb,MutProb,Maximize):
    """
    Implements the Tournament Selection algorithm.
    
    To select each parent select best child of a random 10% of the population.
    """
    import copy 
    
    TournamentSize = int(np.ceil(0.1*len(pop)))
    newPop = np.empty(shape = pop.shape)
    
    currentChild = 0
    while currentChild < pop.shape[0]:
        if Maximize:
            #Parent1
            T1_id = np.random.randint(0,len(fitness), size=TournamentSize)
            T1_fit = np.take_along_axis(fitness, T1_id, axis=0)
            P1_id = T1_id[np.argmax(T1_fit)]
            
            #Parent2
            T2_id = np.random.randint(0,len(fitness), size=TournamentSize)
            T2_fit = np.take_along_axis(fitness, T2_id, axis=0)
            P2_id = T2_id[np.argmax(T2_fit)]
        else:
            #Parent1
            T1_id = np.random.randint(0,len(fitness), size=TournamentSize)
            T1_fit = np.take_along_axis(fitness, T1_id, axis=0)
            P1_id = T1_id[np.argmin(T1_fit)]
            
            #Parent2
            T2_id = np.random.randint(0,len(fitness), size=TournamentSize)
            T2_fit = np.take_along_axis(fitness, T2_id, axis=0)
            P2_id = T2_id[np.argmin(T2_fit)]
        
        if P1_id != P2_id:
            if np.random.random_sample() < CrossProb:
                if np.random.random_sample() < MutProb:
                    newPop[currentChild] = funcMutation(funcCrossover(pop[P1_id],pop[P2_id]))
                else:
                    newPop[currentChild] = funcCrossover(pop[P1_id],pop[P2_id])
            else:
                if fitness[P1_id] > fitness[P2_id]:
                    toCross = copy.deepcopy(pop[P1_id])
                else:
                    toCross = copy.deepcopy(pop[P2_id])
                    
                if np.random.random_sample() < MutProb:
                    newPop[currentChild] = funcMutation(toCross)
                else:
                    newPop[currentChild] = toCross
        
        currentChild += 1
        
    return newPop

def FPSSelection(fitness,pop,funcCrossover,funcMutation,CrossProb,MutProb,Maximize):
    """
    Implements the FPS Selection algorithm.
    
    It computes the probability of each child being chosen based on the fitness score.
    
    Contains elitism funcionality where 50% of the childs are available to selection.
    """
    import copy
    
    Elitism = True
    ElitismPerc = 0.5
    newPop = np.empty(shape = pop.shape)
    
    # To remove Negative Values, sum all elements by the max absolute value 
    # and add a bias to allow even the worst some chance of reproducing
    if Maximize:
        if Elitism:
            SelectionSize = int(np.ceil(fitness.size*ElitismPerc))
            SelectedIDs = np.argpartition(fitness, -SelectionSize)[-SelectionSize:]
            pop = pop[SelectedIDs]
            fitness = fitness[SelectedIDs]
        normalizedfit = np.max(np.abs(fitness) + 100) + fitness
        probs = normalizedfit / np.sum(normalizedfit)
    else:
        if Elitism:
            SelectionSize = int(np.ceil(fitness.size*ElitismPerc))
            SelectedIDs = np.argpartition(fitness*-1.0, -SelectionSize)[-SelectionSize:]
            pop = pop[SelectedIDs]
            fitness = fitness[SelectedIDs]
        normalizedfit = np.max(np.abs(fitness) + 100) + fitness
        reversedfit = np.max(normalizedfit) - normalizedfit
        probs = reversedfit / np.sum(reversedfit)
    
    currentChild = 0
    while currentChild < pop.shape[0]:
        Childs = np.random.choice(np.arange(0,fitness.size), size=2, p=probs, replace = False)
        P1_id = Childs[0]
        P2_id = Childs[1]
        
        if P1_id != P2_id:
            if np.random.random_sample() < CrossProb:
                if np.random.random_sample() < MutProb:
                    newPop[currentChild] = funcMutation(funcCrossover(pop[P1_id],pop[P2_id]))
                else:
                    newPop[currentChild] = funcCrossover(pop[P1_id],pop[P2_id])
            else:
                if fitness[P1_id] > fitness[P2_id]:
                    toCross = copy.deepcopy(pop[P1_id])
                else:
                    toCross = copy.deepcopy(pop[P2_id])
                    
                if np.random.random_sample() < MutProb:
                    newPop[currentChild] = funcMutation(toCross)
                else:
                    newPop[currentChild] = toCross
        
        currentChild += 1
        
    return newPop

def RankSelection(fitness,pop,funcCrossover,funcMutation,CrossProb,MutProb,Maximize):
    """
    Implements the Rank Selection algorithm.
    
    It computes the probability of each child being chosen based on the rank 
    of the fitness score.
    
    Contains elitism funcionality where 50% of the childs are available to selection.
    """
    import copy
    
    Elitism = True
    ElitismPerc = 0.5
    newPop = np.empty(shape = pop.shape)
    
    # To remove Negative Values, sum all elements by the max absolute value 
    # and add a bias to allow even the worst some chance of reproducing
    if Maximize:
        if Elitism:
            SelectionSize = int(np.ceil(fitness.size*ElitismPerc))
            SelectedIDs = np.argpartition(fitness, -SelectionSize)[-SelectionSize:]
            pop = pop[SelectedIDs]
            fitness = fitness[SelectedIDs]
        # Assign ranking to the ids. Highest fitness will have higher rank
        order = fitness.argsort().argsort()
        # Compute probabilities. Highest rank will have highest prob
        probs = (order + 1) / sum(order + 1)
    else:
        if Elitism:
            SelectionSize = int(np.ceil(fitness.size*ElitismPerc))
            SelectedIDs = np.argpartition(fitness*-1.0, -SelectionSize)[-SelectionSize:]
            pop = pop[SelectedIDs]
            fitness = fitness[SelectedIDs]
        # Assign ranking to the ids. Highest fitness will have lower rank
        order = (fitness*-1.0).argsort().argsort()
        # Compute probabilities. Highest rank will have highest prob
        probs = (order + 1) / sum(order + 1)
    
    currentChild = 0
    while currentChild < pop.shape[0]:
        Childs = np.random.choice(np.arange(0,fitness.size), size=2, p=probs, replace = False)
        P1_id = Childs[0]
        P2_id = Childs[1]
        
        if P1_id != P2_id:
            if np.random.random_sample() < CrossProb:
                if np.random.random_sample() < MutProb:
                    newPop[currentChild] = funcMutation(funcCrossover(pop[P1_id],pop[P2_id]))
                else:
                    newPop[currentChild] = funcCrossover(pop[P1_id],pop[P2_id])
            else:
                if fitness[P1_id] > fitness[P2_id]:
                    toCross = copy.deepcopy(pop[P1_id])
                else:
                    toCross = copy.deepcopy(pop[P2_id])
                    
                if np.random.random_sample() < MutProb:
                    newPop[currentChild] = funcMutation(toCross)
                else:
                    newPop[currentChild] = toCross
        
        currentChild += 1
        
    return newPop