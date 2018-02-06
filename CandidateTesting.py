'''
Created on 29 mrt. 2017

@author: sibeleker
'''
import numpy as np
from scipy.optimize import brentq
import math

import functools
import pandas as pd

from ema_workbench import (Model, RealParameter, Constant, load_results, save_results)
from ema_workbench.em_framework import (Scenario, Policy, perform_experiments)
from ema_workbench.em_framework.util import counter, EMAError
from ema_workbench.em_framework.outcomes import AbstractOutcome, ScalarOutcome
from ema_workbench.em_framework import samplers

#THE FUNCTION FOR ANTHROPOGENIC POLLUTION
def a_t(X, #x is a scalar, pollution at time t
        c=[],
        r=[],
        w=[],
        n=2):

    a = sum([w[j]*(abs((X-c[j])/r[j]))**3 for j in range(n)])
    return min(max(a, 0.01), 0.1)


def lake_problem_closedloop(
         b = 0.42,          # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,           # recycling exponent
         mean = 0.02,       # mean of natural inflows
         stdev = 0.001,     # future utility discount rate
         
         delta = 0.98,      # standard deviation of natural inflows
         alpha = 0.4,       # utility from pollution
         nsamples = 100,    # Monte Carlo sampling of natural inflows
         timehorizon = 100, # simulation time,
         seed = 1234,
         **kwargs):         
    '''
    in the closed loop version, utility and inertia are included in the Monte Carlo simulations, too,
    since they are now dependent on X[t].
    '''
    #decisions = [kwargs[str(i)] for i in range(timehorizon)]
    c1 = kwargs['c1']
    c2 = kwargs['c2']
    r1 = kwargs['r1']
    r2 = kwargs['r2']
    w1 = kwargs['w1']
    w2 = 1 - w1
    
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = int(timehorizon)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.zeros((nvars,))

    reliability = 0.0
    utility = 0.0
    inertia = 0.0

    for _ in range(nsamples):
        X[0] = 0.0
        #decisions[0] = a_t(X[0],c=[c1,c2], r=[r1,r2], w=[w1,w2])
        decisions[0] = 0.0
        np.random.seed(seed)
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
  
        for t in range(1,nvars):
            decisions[t] = a_t(X[t-1],c=[c1,c2], r=[r1,r2], w=[w1,w2])
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) +\
                    decisions[t-1] +\
                    natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
        
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
        utility += (np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))) / float(nsamples)
        inertia += (np.sum(np.absolute(np.diff(decisions)) > 0.02)/float(nvars-1)) / float(nsamples)

      
    max_P = np.max(average_daily_P)
    return max_P, utility, inertia, reliability


def lake_problem_openloop(
         b = 0.42,          # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,           # recycling exponent
         mean = 0.02,       # mean of natural inflows
         stdev = 0.001,     # future utility discount rate
         delta = 0.98,      # standard deviation of natural inflows
         
         alpha = 0.4,       # utility from pollution
         nsamples = 100,    # Monte Carlo sampling of natural inflows
         timehorizon = 100, # simulation time
         **kwargs):         

    decisions = [kwargs[str(i)] for i in range(timehorizon)]
    
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = int(timehorizon)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)

    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0

        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
  
        for t in range(1,nvars):
            
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) +\
                    decisions[t-1] +\
                    natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
        
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)

    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    inertia = np.sum(np.absolute(np.diff(decisions)) > 0.02)/float(nvars-1)
      
    max_P = np.max(average_daily_P)
    return max_P, utility, inertia, reliability

if __name__ == "__main__":   

    #instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem_openloop)
    lake_model.time_horizon = 100
    
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]
    
#     lake_model.levers = [RealParameter("c1", -2, 2),
#                          RealParameter("c2", -2, 2),
#                          RealParameter("r1", 0, 2), #[0,2]
#                          RealParameter("r2", 0, 2), #
#                          RealParameter("w1", 0, 1)
#                          ]

    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in 
                     range(lake_model.time_horizon)]

    lake_model.outcomes = [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                           ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                           ScalarOutcome('inertia', kind=ScalarOutcome.MINIMIZE),
                           ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]

    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 100),
                            Constant('timehorizon', lake_model.time_horizon)]
    

    scenarios = ['Ref', 77,  96, 130, 181]
    random_scenarios = [81, 289, 391, 257]
    policies = []
    
    for s in random_scenarios:
#         if s == 'Ref':
#             solutions = pd.DataFrame.from_csv(r'../results/Results_EpsNsgaII_nfe10000_scRef_v3.csv')
#         else:
#             solutions = pd.DataFrame.from_csv(r'../results/Results_EpsNsgaII_nfe10000_sc{}_v5.csv'.format(s))
        
        #checked if there are duplicates: No.
        solutions = pd.DataFrame.from_csv(r'../data/brushed_random_nfe10000_sc{}.csv'.format(s))
        for index, row in solutions.iterrows():
            name = str(s)+'_'+str(index)
            decision = {lever.name:row[lever.name] for lever in lake_model.levers} #levers are in the first columns of the solutions
            policies.append(Policy(name=name, **decision))
    #with MultiprocessingEvaluator(lake_model) as evaluator:
    #    results = evaluator.perform_experiments(scenarios=1000, policies=policies)
    results = perform_experiments(lake_model, 1000, policies, parallel=True)
    save_results(results, r'../CandidateTesting_1000scenarios_revisionRandom_nfe10000.tar.gz')
                                       
