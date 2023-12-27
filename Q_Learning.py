from extended_pd import parallel_env
import numpy as np
import random

def make_QTables(env):
    actions=env.action_space().n
    states=env.observation_space().n
    beginq = tuple(states for _ in env.possible_agents)
    qtable=np.random.uniform(low=0, high=0, size=(beginq+tuple([actions])))
    qtables =[qtable.copy() for _ in env.possible_agents]
    return qtables

def update_QTables(state,action,reward,new_state,qtable,alfa):
    qtable[state+tuple([action])] = (1-alfa)*qtable[state+tuple([action])] + alfa*(reward+0.99*np.max(qtable[new_state]))
    return qtable

def epsilon_greedy(env, qtable, state, epsilon):
    r=random.uniform(0,1)
    action=None
    mean=[]
    if state==None:
        mean=np.mean(qtable[state], axis=1)
    if r>epsilon:
        if state==None:
            action= np.argmax(mean,-1)[0]
        else:
            action= np.argmax(qtable[state],-1)
    else:
        action= env.action_space().sample()
    return action
    
def greedy(qtable,state):
    mean=[]
    if state==None:
        mean=np.mean(qtable[state], axis=1)
        action= np.argmax(mean,-1)[0]
    else:
        action= np.argmax(qtable[state],-1)
    return action

def train(env, n_train_ep, min_epsilon, epsilon, decay, max_steps, qtables):
    alfa=0.01
    adecay=0.99972371
    min_alfa=0.001
    for _ in range(n_train_ep):
        print("-----------------------------------------")
        observations,_=env.reset()
        actions={agent : None for agent in env.possible_agents}
        for agent in env.agents:  
            actions[agent]=env.action_space().sample()
        observations, rewards, terminations, _, _ = env.step(actions)
        for _ in range(max_steps):
            actions={agent : None for agent in env.possible_agents}
            for agent in env.agents:
                state=observations[agent]["state"]
                qtable=qtables[env.agent_name_mapping[agent]]
                if not terminations[agent]:
                    actions.update({agent : epsilon_greedy(env,qtable,tuple(state),epsilon)})
            new_observations, rewards, terminations, _, _ = env.step(actions)
            for agent in env.possible_agents:
                new_state=new_observations[agent]["state"]
                if(actions[agent]!=None):
                    qtable=qtables[env.agent_name_mapping[agent]]
                    qtable=update_QTables(tuple(state),actions[agent],rewards[agent],tuple(new_state),qtable,alfa)
                    qtables[env.agent_name_mapping[agent]]=qtable
            if len(env.agents)==0:
                break
            
            # Our state is the new state
            observations = new_observations
        alfa=max(alfa*adecay, min_alfa)
        epsilon=max(epsilon*decay, min_epsilon)
    return qtables

def evaluate(env, max_steps, n_eval_ep, qtables):
    ep_rewards=[]
    for _ in range(n_eval_ep):
        print("-----------------------------------")
        observations,_=env.reset()
        actions={agent : None for agent in env.possible_agents}
        total_rewards_ep={agent : 0 for agent in env.possible_agents}
        for agent in env.agents:  
            actions[agent]=env.action_space().sample()
        observations, rewards, terminations, _, _ = env.step(actions)
        for agent in env.agents:
            total_rewards_ep[agent]+=rewards[agent]
        
        for _ in range(max_steps):
            actions = {agent : None for agent in env.possible_agents}
            for agent in env.agents:
                qtable=qtables[env.agent_name_mapping[agent]]
                state=observations[agent]["state"]
                if not terminations[agent]:
                    actions.update({agent : greedy(qtable,tuple(state))})
            new_observations, rewards, terminations, _, _ = env.step(actions)
            for agent in env.possible_agents:
                if(actions[agent]!=None):
                    total_rewards_ep[agent]+=rewards[agent]
            if len(env.agents)==0:
                break
            observations=new_observations
        ep_rewards.append(list(total_rewards_ep.values()))
    mean_reward = np.mean(ep_rewards,axis=0)
    std_reward = np.std(ep_rewards,axis=0)
    return mean_reward, std_reward

for _ in range(1):
    env = parallel_env()
    observations, infos = env.reset()
    qtables = make_QTables(env)
    qtables = train(env,10000,0.05,1,0.99964056,100,qtables)
    mean_reward, std_reward = evaluate(env, 100, 100, qtables)
    print(f"Mean_reward={mean_reward[0]:.2f} +/- {std_reward[0]:.2f}")
    print(f"Mean_reward={mean_reward[1]:.2f} +/- {std_reward[1]:.2f}")
    print(f"Mean_reward={mean_reward[2]:.2f} +/- {std_reward[2]:.2f}")
    print(qtables)
    break