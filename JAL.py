from extended_pd import parallel_env
import numpy as np
import random

## QTABLE SIZE PER AGENT: NUM_STATES^NUM_AGENTS x NUM_ACTIONS^NUM_AGENTS
def make_QTables(env, gamma):
    actions=env.action_space().n
    states=env.observation_space().n
    begin = tuple(states for _ in env.possible_agents)
    max_rew=((env.utility_per_agent*(len(env.possible_agents)-1))/len(env.possible_agents))/((actions-1)*(1-env.k))
    end = tuple(actions for _ in env.agents)
    qtable=np.full((begin+end),max_rew)
    qtables =[qtable.copy() for _ in env.possible_agents]
    return qtables

## ACTION COUNT TABLE SIZE PER AGENT: NUM_STATES^NUM_AGENTS x NUM_ACTIONS^NUM_AGENTS-1 (count other agents' actions)
## STATE COUNT TABLE SIZE: 1 x NUM_STATES^NUM_AGENTS (count times state is visited)
def make_Counts(env):
    actions=env.action_space().n
    states=env.observation_space().n
    begin = tuple(states for _ in env.possible_agents)
    end = tuple(actions for _ in range(len(env.possible_agents)-1))
    actionCount=np.zeros(begin+end)
    stateCounts=np.zeros(begin)
    actionCounts=[actionCount.copy() for _ in env.possible_agents]
    return actionCounts,stateCounts

def make_CountTables(env):
    actions=env.action_space().n
    states=env.observation_space().n
    beginq = tuple(states for _ in env.possible_agents)
    count=np.zeros((beginq+tuple([actions])))
    counts =[count.copy() for _ in env.possible_agents]
    return counts

## QTABLE UPDATE RULE:
## Q(s,a) = (1-lr)*Q(s,a) + lr*(r + dr*MAX(Q(s',a')))
## lr = alfa = learning rate
## dr = gamma = discount rate
## r = reward obtained when taking action a in state s

#To find best: take assumed probabilities into account (i.e. weigh by action/state)
def update_QTables(states,actions,reward,new_states,qtable,alfa,actionCounts,stateCounts,gamma):
    if(stateCounts!=0):
        weights=actionCounts/stateCounts
    else:
        weights=actionCounts

    if(len(weights.shape)>1):
        for i in range(len(weights)):
            if(np.sum(weights[i])==0):
                weights[i]=[1/len(weights[i]) for _ in weights[i]]
    else:
        if(np.sum(weights,axis=-1)==0):
            weights=[1/len(weights) for _ in weights]
    
    weighted_q=(qtable[new_states].T*weights).T
    total=np.sum(weighted_q,axis=0)
    while len(total.shape)>1:
        total=np.sum(total,axis=0)/total.shape[0]
    best=np.max(total)
    qtable[states][actions]=((1-alfa)*qtable[states][actions]+alfa*(reward+gamma*best))
    return qtable

#To find best: take assumed probabilities into account (i.e. weigh by action/state)
def epsilon_greedy(env, qtable, agent, states, epsilon, actionCounts, stateCounts):
    r=random.uniform(0,1)
    action=None
    if(stateCounts[states]!=0):
        weights=actionCounts/stateCounts[states]
    else:
        weights=actionCounts
    if(len(weights.shape)>1):
        for i in range(len(weights)):
            if(np.sum(weights[i])==0):
                weights[i]=[1/len(weights[i]) for _ in weights[i]]
    else:
        if(np.sum(weights,axis=-1)==0):
            weights=[1/len(weights) for _ in weights]
    weighted_q=(qtable[states].T*weights).T
    total=np.sum(weighted_q,axis=0)
    while len(total.shape)>1:
        total=np.sum(total,axis=0)/total.shape[0]
    best=np.argmax(total,axis=-1)  
    if r>epsilon:
        action=best
    else:
        action=env.action_space(agent).sample()
    return action
    
#To find best: take assumed probabilities into account (i.e. weigh by action/state)
def greedy(qtable,states, actionCounts, stateCounts):
    if(stateCounts[states]!=0):
        weights=actionCounts/stateCounts[states]
    else:
        weights=actionCounts
    if(len(weights.shape)>1):
        for i in range(len(weights)):
            if(np.sum(weights[i])==0):
                weights[i]=[1/len(weights[i]) for _ in weights[i]]
    else:
        if(np.sum(weights,axis=-1)==0):
            weights=[1/len(weights) for _ in weights]
    weighted_q=(qtable[states].T*weights).T
    total=np.sum(weighted_q,axis=0)
    while len(total.shape)>1:
        total=np.sum(total,axis=0)/total.shape[0]
    
    best=np.argmax(total) 
    return best

def train(env, n_train_ep, min_epsilon, epsilon, decay, max_steps, qtables, stateCounts, actionCounts, gamma, alfa, adecay):
    train_rewards=[]
    counts=make_CountTables(env)
    for _ in range(n_train_ep):
        t_rewards={agent : 0 for agent in env.possible_agents}
        observations,_=env.reset()
        actions={agent : None for agent in env.possible_agents}
        for agent in env.agents:  
            actions[agent]=env.action_space().sample()
        observations, rewards, terminations, _, _ = env.step(actions)
        for agent in env.agents:
            t_rewards[agent]+=rewards[agent]
        for k in range(max_steps):
            state=observations[env.agents[0]]["state"]
            stateCounts[state]+=1
            actions={agent : None for agent in env.possible_agents}

            for agent in env.agents:
                qtable=qtables[env.agent_name_mapping[agent]]
                if not terminations[agent]:
                    actions.update({agent : epsilon_greedy(env,qtable,agent,tuple(state),epsilon,actionCounts[env.agent_name_mapping[agent]][tuple(state)],stateCounts)})
            new_observations, rewards, terminations, _, _ = env.step(actions)
            if len(env.agents)==0:
                break
            new_state=new_observations[env.agents[0]]["state"]
            for agent in env.possible_agents:
                if(actions[agent]!=None):
                    id=env.agent_name_mapping[agent]
                    counts[id][tuple(state)][actions[agent]]+=1
                    t_rewards[agent]+=rewards[agent]
                    qtable=qtables[id]
                    other_actions=tuple([actions[a] for a in env.agents if a!=agent])
                    my_action=tuple([actions[agent]])
                    count=counts[id][tuple(state)][actions[agent]]
                    learning_rate=alfa/(1+adecay*count)
                    qtable=update_QTables(tuple(state),other_actions+my_action,rewards[agent],tuple(new_state),qtable,learning_rate,actionCounts[id][tuple(new_state)],stateCounts[tuple(new_state)],gamma)
                    toChange=[list(other_actions)]
                    actionCounts[id][tuple(state)][tuple(toChange)]+=1
                    qtables[id]=qtable
            
            # Our state is the new state
            observations = new_observations
        epsilon=max(epsilon-decay, min_epsilon)
        train_rewards.append(np.mean([i/max_steps for i in list(t_rewards.values())]))
    return qtables,train_rewards


""" for _ in range(1):
    gamma=0.95
    alfa=0.01
    adecay=0.0001
    env = parallel_env()
    observations, infos = env.reset()
    qtables = make_QTables(env,gamma)
    actionCounts,stateCounts= make_Counts(env)
    qtables,tot_rew = train(env,40000,0,0.2,0.000006,100,qtables,stateCounts,actionCounts,gamma,alfa,adecay)
    print(tot_rew)
    break
    mean_reward, std_reward = evaluate(env, 100, 100, qtables)
    print(f"Mean_reward={mean_reward[0]:.2f} +/- {std_reward[0]:.2f}")
    print(f"Mean_reward={mean_reward[1]:.2f} +/- {std_reward[1]:.2f}")
    print(f"Mean_reward={mean_reward[2]:.2f} +/- {std_reward[2]:.2f}")
    break """