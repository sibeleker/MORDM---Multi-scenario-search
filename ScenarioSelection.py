'''
Created on 20 mrt. 2017

@author: sibeleker
'''
import os
import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist 
from ema_workbench import load_results
from functools import partial
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd

from ema_workbench.util.utilities import load_results
def normalize_out_dic(outcomes):
    norm_outcomes = {}
    for ooi in outcomes.keys():
        data = outcomes[ooi]
        mx = max(data)
        mn = min(data)
        if mx == mn:
            norm_outcomes[ooi] = data - mn
        else:
            norm_outcomes[ooi] = (data - mn)/(mx-mn)
    return norm_outcomes

def calculate_distance(data, oois, scenarios=None, distance='euclidean'):
    '''data is the outcomes of exploration results,
    scenarios is a list of scenario indices (decision variables), 
    oois is a list of variable names,
    distance is to choose the distance metric. options:
            bray-curtis, canberra, chebyshev, cityblock (manhattan), correlation, 
            cosine, euclidian, mahalanobis, minkowski, seuclidian,
            sqeuclidian, wminkowski
    returns a list of distance values
    '''
    #make a matrix of the data n_scenarios x oois
    scenario_data = np.zeros((len(scenarios), len(oois)))
    for i, s in enumerate(scenarios):
        for j, ooi in enumerate(oois):
            scenario_data[i][j] = data[ooi][s]
                
    distances = pdist(scenario_data, distance)
    return distances

def evaluate_diversity_single(x, data, oois, weight, distance):
    '''
    takes the outcomes and selected scenario set (decision variables), 
    returns a single 'diversity' value for the scenario set.
    outcomes : outcomes dictionary of the scenario ensemble
    decision vars : indices of the scenario set
    weight : weight given to the mean in the diversity metric. If 0, only minimum; if 1, only mean
    '''
    distances = calculate_distance(data, oois, x, distance)
    minimum = np.min(distances)
    mean = np.mean(distances)
    diversity = (1-weight)*minimum + weight*mean
    
    return [diversity]

def find_maxdiverse_scenarios(outcomes, set_size, weight, distance):
    oois = list(outcomes.keys())
    n_scen = len(outcomes[oois[0]])
    indices = range(n_scen)
    diversity = 0.0
    solutions = []
    for sc_set in itertools.combinations(indices, set_size):
        temp_div = evaluate_diversity_single(list(sc_set), outcomes, oois, weight, distance)
        if temp_div[0] > diversity:
            diversity = temp_div[0]
            solutions = []
            solutions.append(sc_set)
        elif temp_div[0] == diversity:
            solutions.append(sc_set)
    return diversity, solutions

def find_policy_relevant_scenarios(results):
    experiments, outcomes = results
    oois = sorted(outcomes.keys())
    indices = []
    for ooi in oois:
        if ooi == 'max_P':
            a = outcomes[ooi] >= np.median(outcomes[ooi])     
        else: 
            a = outcomes[ooi] <= np.median(outcomes[ooi])
        indices.append(a)
    indices = np.swapaxes(indices, 0, 1)
    logical_index = np.array([index.all() for index in indices])
    newExperiments = experiments[logical_index]
    newOutcomes = {}
    for ooi in oois:
        newOutcomes[ooi] = outcomes[ooi][logical_index]
    newResults = newExperiments, newOutcomes
    return newResults


if __name__ == "__main__":
    dir = 'D:/sibeleker/surfdrive/Documents/Notebooks/Lake_model-MORDM/Sibel/data/'
    fn = '1000_experiments_closedloop_noApollution_correctedInertia.tar.gz'
    results = load_results(dir+fn)
    newResults = find_policy_relevant_scenarios(results)
    exp, outcomes = newResults
    print(len(outcomes['max_P']))
    
    norm_new_out = normalize_out_dic(outcomes)
    
    distances = ['euclidean', 'cityblock']
    weights = np.arange(0, 1.05, 0.25)
    print(weights)
    # empty df to be filled in
    cols = ['distance', 'weight', 'selected', 'diversity']
    rows = range(len(distances)*len(weights))
    df = pd.DataFrame(columns=cols, index=rows)
    
    
    pool = multiprocessing.Pool(processes=4)

    timeout = 8000
    
    index = 0

    for distance in distances:
        for weight in weights:
            df['distance'][index] = distance
            df['weight'][index] = weight
            start = time.time()
            result = pool.apply_async(find_maxdiverse_scenarios, args=(norm_new_out, 4, weight, distance))
            try:
                diversity, selected = result.get(timeout)
                df['selected'][index] = selected
                df['diversity'][index] = diversity
            except TimeoutError:
                print("TimeoutError")
            

            end = time.time()
            index += 1
            df.to_csv('./Results_Scenario_selection_{}.csv'.format(index))
            print("iteration {} took {} seconds.".format(index, end-start))
    #df.to_csv('./Results_Scenario_selection_v2.csv')
    
        
    pool.close()
    pool.join()