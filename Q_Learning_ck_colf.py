from extended_pd import parallel_env
import numpy as np
import random



def makeEmptyTables(env):
    actions=env.action_space().n
    states=env.observation_space().n
    q_dim = tuple(states for _ in env.possible_agents)
    empty_table=np.zeros(q_dim + tuple([actions]))
    empty_tables = [empty_table.copy() for _ in env.possible_agents]
    return empty_tables
 

def make_QTables(env,gamma):
    actions=env.action_space().n
    states=env.observation_space().n
    beginq = tuple(states for _ in env.possible_agents)
    max_rew=((env.utility_per_agent*(len(env.possible_agents)-1))/len(env.possible_agents))/((actions-1)*(1-env.k))
    qtable=np.full((beginq+tuple([actions])), max_rew/(1-gamma))
    qtables =[qtable.copy() for _ in env.possible_agents]
    return qtables

def make_CountTables(env):
    actions=env.action_space().n
    states=env.observation_space().n
    beginq = tuple(states for _ in env.possible_agents)
    count=np.zeros((beginq+tuple([actions])))
    counts =[count.copy() for _ in env.possible_agents]
    return counts


def calculate_correct_learning_rate(reward,action,state,ptable,stable,alfa,adecay,count,non_stationary_multiplier):
    abs_difference = abs(reward - ptable[tuple(state)+tuple([action])])
    alfa_non_stationary = alfa
    alfa_stationary = non_stationary_multiplier*alfa

    if abs_difference > stable[tuple(state)+tuple([action])]:
        learning_rate = alfa_non_stationary/(1+adecay*count)
    else:
        learning_rate = alfa_stationary/(1+adecay*count)
    return abs_difference,learning_rate




def update_PTables(state,action,ptable,reward,lamb):
    idx = state+tuple([action])
    ptable[idx] = (1-lamb)*ptable[idx] + reward*lamb
    return ptable



def update_STables(state,action,stable,abs_difference_reward,lamb):
    idx = state+tuple([action])
    stable[idx] = (1-lamb)*stable[idx] + abs_difference_reward*lamb
    return stable
 

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

def train(env, n_train_ep, min_epsilon, epsilon, decay, max_steps, qtables,ptables,stables,gamma, alfa, adecay,non_stationary_multiplier,lamb):
    # Keep: False
    # Update: True 
    total_rewards=[]
    status_per_agent = {a: True for a in env.possible_agents}
    a_update_per_agent = {a: None for a in env.possible_agents}
    prev_action_per_agent = a_update_per_agent = {a: None for a in env.possible_agents}
    counts=make_CountTables(env)
    for _ in range(n_train_ep):
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


            new_observations, rewards, terminations, _, _ = env.step(actions)
            for agent in env.possible_agents:
                ptable = ptables[env.agent_name_mapping[agent]]
                stable = stables[env.agent_name_mapping[agent]]
                qtable = qtables[env.agent_name_mapping[agent]]

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
                            reward = rewards[agent]
                            action = actions[agent]


                            status_per_agent[agent] = True
                            qtable=qtables[env.agent_name_mapping[agent]]
                            count=counts[env.agent_name_mapping[agent]][tuple(state)][actions[agent]]

                            abs_diff,learning_rate = calculate_correct_learning_rate(reward,action,state,ptable,stable,alfa,adecay,count,non_stationary_multiplier)
                            qtable=update_QTables(tuple(state),actions[agent],rewards[agent],tuple(new_state),qtable,learning_rate,gamma)
                            ptable = update_PTables(tuple(state),actions[agent],ptable,rewards[agent],lamb)
                            stable = update_STables(tuple(state),actions[agent],stable,abs_diff,lamb)

                            qtables[env.agent_name_mapping[agent]]=qtable
                    #Re-use old state and give agents time to respond to action change
                    else:
                        reward = rewards[agent]
                        action = actions[agent]
                        abs_diff,learning_rate = calculate_correct_learning_rate(reward,action,state,ptable,stable,alfa,adecay,count)
                        qtable=qtables[env.agent_name_mapping[agent]]
                        count=counts[env.agent_name_mapping[agent]][tuple(state)][actions[agent]]
                        learning_rate=alfa/(1+adecay*count)
                        prev_state = tuple(a_update_per_agent[agent])
                        ptable = update_PTables(tuple(state),actions[agent],ptable,rewards[agent],lamb)
                        stable = update_STables(tuple(state),actions[agent],stable,abs_diff,lamb)
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

#for _ in range(1):
#    gamma=0.95
#    alfa=0.01
#    adecay=0.0001
#    lamb = 0.1
#    env = parallel_env()
#    observations, infos = env.reset()
#    qtables = make_QTables(env,gamma)
#    stables = makeEmptyTables(env)
#    ptables = makeEmptyTables(env)
#    qtables,total_rewards = train(env,10000,0,0.2,0.000006,100,qtables,ptables,stables,gamma,alfa,adecay,4,lamb)
#    print(total_rewards)
#    break
#    mean_reward, std_reward = evaluate(env, 100, 100, qtables)
#    print(f"Mean_reward={mean_reward[0]:.2f} +/- {std_reward[0]:.2f}")
#    print(f"Mean_reward={mean_reward[1]:.2f} +/- {std_reward[1]:.2f}")
#    print(f"Mean_reward={mean_reward[2]:.2f} +/- {std_reward[2]:.2f}")
#   print(qtables)
#   break