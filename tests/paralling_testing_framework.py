import sys
import multiprocessing
# setting path
sys.path.append('..')
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from Q_Learning import train as train_default, make_QTables as make_QTables_default
from Q_Learning_ck import train as train_ck, make_QTables as make_QTables_ck
from Q_Learning_colf import train as train_colf, make_QTables as make_QTables_colf, makeEmptyTables as makeEmptyTables_colf
from Q_Learning_ck_colf import train as train_ck_colf, make_QTables as make_QTables_ck_colf, makeEmptyTables as makeEmptyTables_ck_colf

from extended_pd import parallel_env
# Alpha colf test
trials = 1

implementations = [
    "Q_Learning",
    "Q_Learning_ck",
    "Q_Learning_colf",
    "Q_Learning_CK_COLF"
]


params={
     
    "alfa"          : 0.1     , 
    "gamma"         : 0.95     ,
    "adecay"        : 0.0001   ,  
    "lamb"          : 0.1      ,  
    "n_episode"     : 10000,  
    "epsilon_decay" : 0.00006  , 
    "max_steps"     : 100      ,
    "k"             : 2/3      

     
    }

def run_implementations(params):
       alfa          = params["alfa"                      ]
       gamma         = params["gamma"                     ]
       adecay        = params["adecay"                    ]
       lamb          = params["lamb"                      ]
       n_episode     = params["n_episode"                 ]
       epsilon_decay = params["epsilon_decay"             ]
       max_steps     = params["max_steps"                 ]
       implementation= params["implementation"            ]
       k             = params["k"]

       if (implementation == "Q_Learning"):
           env = parallel_env(k=k)
           observations, infos = env.reset()
           qtables_Q = make_QTables_default(env, gamma)
           qtables_Q, tot_rew_per_gamma = train_default(env, n_episode, 0, 0.2, epsilon_decay, max_steps, qtables_Q, gamma, alfa, adecay)
           return tot_rew_per_gamma
           # tot_rew_per_episode.append(tot_rew_per_gamma)
       
       elif (implementation == "Q_Learning_ck"):
           env = parallel_env(k=k)
           observations, infos = env.reset()
           qtables_ck = make_QTables_ck(env, gamma)
           qtables_ck, tot_rew_per_gamma = train_ck(env, n_episode, 0, 0.2, epsilon_decay, max_steps, qtables_ck, gamma, alfa, adecay)
           return tot_rew_per_gamma
           # tot_rew_per_episode.append(tot_rew_per_gamma)

       elif (implementation == "Q_Learning_colf"):
           env = parallel_env(k=k)
           observations, infos = env.reset()
           qtables_colf = make_QTables_colf(env, gamma)
           stables_colf = makeEmptyTables_colf(env)
           ptables_colf = makeEmptyTables_colf(env)
           qtables_colf, tot_rew_per_gamma = train_colf(env, n_episode, 0, 0.2, epsilon_decay, max_steps, qtables_colf, ptables_colf, stables_colf, gamma, alfa, adecay, 4, lamb)
           return tot_rew_per_gamma
           # tot_rew_per_episode.append(tot_rew_per_gamma)

       elif (implementation == "Q_Learning_CK_COLF"):
           env = parallel_env(k=k)
           observations, infos = env.reset()
           qtables_colf_ck = make_QTables_ck_colf(env, gamma)
           stables_colf_ck = makeEmptyTables_ck_colf(env)
           ptables_colf_ck = makeEmptyTables_ck_colf(env)
           qtables_colf_ck, tot_rew_per_gamma = train_ck_colf(env, n_episode, 0, 0.2, epsilon_decay, max_steps, qtables_colf_ck, ptables_colf_ck, stables_colf_ck, gamma, alfa, adecay, 4, lamb)
           return tot_rew_per_gamma


def run_parallel(params):
    pool = multiprocessing.Pool(processes=5)
    total_rewards = pool.map(run_implementations,params)
    return total_rewards



# This was just for alfas: example

# params_Q_alfa_04 = params.copy()
# params_Q_alfa_04["alfa"] = 0.4  
# params_Q_alfa_04["implementation"] = "Q_Learning"  
# params_Q_alfa_01 = params.copy()
# params_Q_alfa_01["alfa"] = 0.1  
# params_Q_alfa_01["implementation"] = "Q_Learning"  
# params_Q_colf = params.copy()
# params_Q_colf["implementation"] = "Q_Learning_colf"

# _params = [params_Q_alfa_01,params_Q_alfa_04,params_Q_colf]



# print(run_parallel(_params))



    
    
    
    