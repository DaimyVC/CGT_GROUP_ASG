from extended_pd import parallel_env
import numpy as np
import random
from tqdm import tqdm

def make_QTables(env,gamma):
    actions=env.action_space().n
    states=env.observation_space().n
    beginq = tuple(states for _ in env.possible_agents)
    max_rew=((env.utility_per_agent*(len(env.possible_agents)-1))/len(env.possible_agents))/((actions-1)*(1-env.k))
    qtable=np.full((beginq+tuple([actions])), max_rew/gamma)
    qtables =[qtable.copy() for _ in env.possible_agents]
    return qtables

def make_CountTables(env):
    actions=env.action_space().n
    states=env.observation_space().n
    beginq = tuple(states for _ in env.possible_agents)
    count=np.zeros((beginq+tuple([actions])))
    counts =[count.copy() for _ in env.possible_agents]
    return counts

def update_QTables(state,action,reward,new_state,qtable,alfa,gamma):
    qtable[state+tuple([action])] = (1-alfa)*qtable[state+tuple([action])] + alfa*(reward+gamma*np.max(qtable[new_state]))
    
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

def train(env, n_train_ep, min_epsilon, epsilon, decay, max_steps, qtables,gamma, alfa, adecay):
    # Keep: False
    # Update: True 
    status_per_agent = {a: True for a in env.possible_agents}
    a_update_per_agent = {a: None for a in env.possible_agents}
    total_rewards=[]
    prev_action_per_agent = {a: None for a in env.possible_agents}
    counts=make_CountTables(env)
    for _ in tqdm(range(n_train_ep)):
        ep_rewards={agent:0 for agent in env.possible_agents}
        # print("-----------------------------------------")
        observations,_=env.reset()
        actions={agent : None for agent in env.possible_agents}
        for agent in env.agents:  
            actions[agent]=env.action_space().sample()
        observations, rewards, terminations, _, _ = env.step(actions)
        for agent in env.agents:
            ep_rewards[agent]+=rewards[agent]
        for _ in range(max_steps):
            actions={agent : None for agent in env.possible_agents}
            for agent in env.agents:
                state=observations[agent]["state"]
                qtable=qtables[env.agent_name_mapping[agent]]
                if not terminations[agent]:
                    
                    ## Change action when on update mode otherwise keep the same as previous

                    if status_per_agent[agent]:
                        actions.update({agent : epsilon_greedy(env,qtable,tuple(state),epsilon)})
                        counts[env.agent_name_mapping[agent]][tuple(state)][actions[agent]]+=1
                    else:
                        actions[agent] = prev_action_per_agent[agent]
                prev_action_per_agent[agent] = actions[agent]


            # print(actions)
            new_observations, rewards, terminations, _, _ = env.step(actions)
            for agent in env.possible_agents:
                new_state=new_observations[agent]["state"]
                if(actions[agent]!=None):
                    ep_rewards[agent]+=rewards[agent]
                    
                    if status_per_agent[agent]:
                        # Save previous state for next iteration
                        if actions[agent] != prev_action_per_agent[agent]:
                            status_per_agent[agent] = False
                            a_update_per_agent[agent] = state
                        # Update with new state 
                        else: 
                            status_per_agent[agent] = True
                            qtable=qtables[env.agent_name_mapping[agent]]
                            count=counts[env.agent_name_mapping[agent]][tuple(state)][actions[agent]]
                            learning_rate=alfa/(1+adecay*count)
                            qtable=update_QTables(tuple(state),actions[agent],rewards[agent],tuple(new_state),qtable,learning_rate,gamma)
                            qtables[env.agent_name_mapping[agent]]=qtable
                    #Re-use old state and give agents time to respond to action change
                    else:
                        qtable=qtables[env.agent_name_mapping[agent]]
                        count=counts[env.agent_name_mapping[agent]][tuple(state)][actions[agent]]
                        learning_rate=alfa/(1+adecay*count)
                        prev_state = tuple(a_update_per_agent[agent])
                        qtable=update_QTables(prev_state,actions[agent],rewards[agent],tuple(new_state),qtable,learning_rate,gamma)
                        qtables[env.agent_name_mapping[agent]]=qtable




                    
            if len(env.agents)==0:
                break
            
            # Our state is the new state
            observations = new_observations
        epsilon=max(epsilon-decay, min_epsilon)
        total_rewards.append(np.mean([i/max_steps for i in list(ep_rewards.values())]))
    return qtables,total_rewards

def evaluate(env, max_steps, n_eval_ep, qtables):
    ep_rewards=[]
    for _ in range(n_eval_ep):
        # print("-----------------------------------")
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

# for _ in range(1):
#    gamma=0.95
#    alfa=0.1
#    adecay=0.0001
#    env = parallel_env()
#    observations, infos = env.reset()
#    qtables = make_QTables(env,gamma)
#    qtables,total_reward = train(env,1000,0.1,0.2,0.00006,1000,qtables,gamma,alfa,adecay)
#    print(total_reward)
#    break
#    mean_reward, std_reward = evaluate(env, 100, 100, qtables)
#    print(f"Mean_reward={mean_reward[0]:.2f} +/- {std_reward[0]:.2f}")
#    print(f"Mean_reward={mean_reward[1]:.2f} +/- {std_reward[1]:.2f}")
#    print(f"Mean_reward={mean_reward[2]:.2f} +/- {std_reward[2]:.2f}")
#    print(qtables)
#    break