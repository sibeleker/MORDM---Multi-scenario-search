'''
Created on 13 mrt. 2017

@author: sibeleker
'''

import numpy as np
from scipy.optimize import brentq
import math
import matplotlib.pyplot as plt
import multiprocessing
import time
from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, Real, EpsNSGAII, Hypervolume, calculate

import functools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle

from ema_workbench import (Model, RealParameter, Constant, load_results)
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
         timehorizon = 100, # simulation time
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

def evaluate_function(x, scenario={}, model=None, decision_vars=None, searchover=None):
    '''helper function for transforming decision variables to correctly
    formatted input for running the model
    
    Parameters
    ----------
    x : list
    model : a Model instance
    decision_vars : list of Parameter instances 
    searchover : {'uncertainties', 'levers'}
    
    note:: model and decision_vars are preloaded through functools.partial
    
    TODO:: currently only handles outcomes, should also handle constraints
    
    '''

    # we need to link the decision variables with the name of 
    # each decision variable
    decision = {lever.name:x[i] for i, lever in enumerate(decision_vars)}

    # we can now evaluate the model
    if searchover=='levers':
        model.run_model(Scenario(scenario), Policy(name=counter, **decision))
    else:
        model.run_model(Scenario(**decision), Policy({}))
    
    result = model.output
    outcomes = [result[o.name] for o in model.outcomes  
                if o.kind != AbstractOutcome.INFO]
    
    return outcomes

def plot_convergence(fe_results, scenario):
    '''
    fe_results is function evaluation results, 2d list, iterations x 2 (0: nfe, 1: hypervolume)
    '''
    
    # if myltiple objectives
     
#     no_obj = results.shape[0]
#     figure, axes = plt.subplots(no_obj, 1)
#     figure.set_figheight(15)
#     figure.set_figwidth(9) 
#     
#     for i in range(no_obj):
#         #ax = figure.add_subplot(grid[i,0])
#         ax = axes[i]
#         ax.plot(results[i])
#         ax.set_title("Convergence for objective {}".format(i), fontsize=16)
#         ax.set_xlabel("nfe")
    
    # if single value for hypervolume
    results = np.swapaxes(np.array(fe_results), 0, 1)
    nfe = results[0]
    hv = results[1]
    figure = plt.figure()
    sns.set_style("whitegrid")
    ax = figure.add_subplot(111)
    ax.plot(nfe, hv, color='black')
    ax.set_xlabel("NFE", fontsize=12)
    ax.set_ylabel("Hypervolume", fontsize=12)
    ax.set_title("Convergence of the search process", fontsize=16)
    
    
    plt.savefig('./Convergence_EpsNsgaII_nfe10000_sd1234_epsforeach_s{}.png'.format(scenario))     
    plt.show()

def optimize(model, scenario, nfe, epsilons, sc_name, algorithm=EpsNSGAII, searchover='levers'):
    '''optimize the model
    
    Parameters
    ----------
    model : a Model instance
    algorith : a valid Platypus optimization algorithm
    nfe : int
    searchover : {'uncertainties', 'levers'}
    
    Returns
    -------
    pandas DataFrame
    
    
    Raises
    ------
    EMAError if searchover is not one of 'uncertainties' or 'levers'
    
    TODO:: constrains are not yet supported
    
    '''
    if searchover not in ('levers', 'uncertainties'):
        raise EMAError(("searchover should be one of 'levers' or"
                        "'uncertainties' not {}".format(searchover)))
    
    # extract the levers and the outcomes
    decision_variables = [dv for dv in getattr(model, searchover)]
    outcomes = [outcome for outcome in model.outcomes if 
                outcome.kind != AbstractOutcome.INFO]
    
    evalfunc = functools.partial(evaluate_function, model=model,
                                 scenario=scenario,
                                 decision_vars=decision_variables,
                                 searchover=searchover)
    
    # setup the optimization problem
    # TODO:: add constraints
    problem = Problem(len(decision_variables), len(outcomes))
    problem.types[:] = [Real(dv.lower_bound, dv.upper_bound) 
                        for dv in decision_variables]
    problem.function = evalfunc
    problem.directions = [outcome.kind for outcome in outcomes]

    # solve the optimization problem
    optimizer = algorithm(problem, epsilons=epsilons)
    optimizer.run(nfe)

    # extract the names for levers and the outcomes
    lever_names = [dv.name for dv in decision_variables]
    outcome_names = [outcome.name for outcome in outcomes]
    
    solutions = []
    for solution in unique(nondominated(optimizer.result)):
        decision_vars = dict(zip(lever_names, solution.variables))
        decision_out = dict(zip(outcome_names, solution.objectives))
        result = {**decision_vars, **decision_out} 
        solutions.append(result)
    
    #print("fe_result: ", optimizer.algorithm.fe_results)
    #plot_convergence(optimizer.algorithm.hv_results, sc_name)
    results = pd.DataFrame(solutions, columns=lever_names+outcome_names)
    
    #save the hypervolume output in a csv file
    hv = np.swapaxes(np.array(optimizer.algorithm.hv_results), 0, 1) #hv is a 2d list, where hv[0] is the record of nfe's, hv[1] is the record of hypervolume
    df = pd.DataFrame(hv).transpose()
    #df.to_csv("Hypervolume_scenario_{}_v6.csv".format(sc_name))
    
    return results, df


if __name__ == "__main__":   

    #instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem_closedloop)
    lake_model.time_horizon = 100
    #specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]
    
    # set levers, one for each time step
    lake_model.levers = [RealParameter("c1", -2, 2),
                         RealParameter("c2", -2, 2),
                         RealParameter("r1", 0, 2), #[0,2]
                         RealParameter("r2", 0, 2), #
                         RealParameter("w1", 0, 1)
                         ]
    
    #specify outcomes 
    lake_model.outcomes = [ScalarOutcome('max_P', kind=ScalarOutcome.MINIMIZE),
                           ScalarOutcome('utility', kind=ScalarOutcome.MAXIMIZE),
                           ScalarOutcome('inertia', kind=ScalarOutcome.MINIMIZE),
                           ScalarOutcome('reliability', kind=ScalarOutcome.MAXIMIZE)]
    
    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 100),
                            Constant('timehorizon', lake_model.time_horizon),
                           ]
    #Load the initial exploration for scenarios
    directory = 'D:/sibeleker/surfdrive/Documents/Notebooks/Lake_model-MORDM/Sibel/data/'
    fn = '211_experiments_closedloop_noApollution_inertia.tar.gz'
    scenario_results = load_results(directory+fn)
    experiments, outcomes = scenario_results
    #get the selected scenarios
    scenarios = ['Ref', 153, 160, 197, 207] #EUCLIDEAN, W=0.5
    #optimize for each selected scenario
    pool = multiprocessing.Pool(processes=3)
    timeout = 3000
    start = time.time()

    
    for s in scenarios:
        if s == 'Ref':
            result = pool.apply_async(optimize, args=(lake_model, {}, 5000, [0.1, 0.05, 0.005, 0.005], 'scRef'))
        else:
            scenario = experiments[s]
            result = pool.apply_async(optimize, args=(lake_model, scenario, 5000, [0.1, 0.05, 0.005, 0.005], 'sc{}'.format(s)))
        try:
            results, df = result.get(timeout)

        except TimeoutError:
            print("TimeoutError")
            

        end = time.time()
        print("scenario {} took {} seconds.".format(s, end-start))
        
        fn1 = './Results_EpsNsgaII_nfe10000_sc{}_v7.csv'.format(s)
        fn2 = './Hypervolume_scenario_sc{}_v7.csv'.format(s)
        results.to_csv(fn1)
        df.to_csv(fn2)
        print('found {} solutions for scenario {}'.format(results.values.shape[0], s))


