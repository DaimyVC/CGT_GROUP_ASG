import functools
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete,MultiDiscrete
import random

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

NUM_ITERS=10**6


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "env_1"}

    def __init__(self, render_mode=None, num_agents=3, num_moves=3, k=1/3):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["player_" + str(r) for r in range(1,num_agents+1)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.utility_per_agent=num_moves
        self.render_mode = render_mode
        self.k=k

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self,agent=None):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return Discrete(self.utility_per_agent+1)


    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self,agent=None):
            return Discrete(self.utility_per_agent+1)

    #TODO
    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        """Renders the environment."""
        

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.a_update = None 
        self.num_moves=0
        self.agents = self.possible_agents[:]
        self.agent_utility =[self.utility_per_agent for _ in self.possible_agents]
        observations = {agent: {} for agent in self.agents}
        for agent in self.agents:
            observations[agent].update({"state":[None for a in self.possible_agents]})
        infos = {agent: {} for agent in self.agents}
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # rewards for all agents are placed in the rewards dictionary to be returned
        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        rew = np.zeros((len(self.possible_agents)))

        step_actions=[None for _ in self.possible_agents]
        for agent in self.possible_agents:
            if agent in self.agents:
                step_actions[self.agent_name_mapping[agent]]=actions[agent]

        for agent in self.agents:
            id=self.agent_name_mapping[agent]
            rew[id]=(np.mean(step_actions)-self.k*(step_actions[id]))/(self.utility_per_agent*(1-self.k))

        rewards = {agent : 0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: env_truncation for agent in self.possible_agents}
        observations = {agent: {} for agent in self.possible_agents}

        if 0 not in list(terminations.values()):
            self.agents = []

        # current observation is just the players most recent actions
        for agent in self.possible_agents:
            rewards.update({agent:rew[self.agent_name_mapping[agent]]})
            observations[agent].update({"state":[a for a in step_actions]})

        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        return observations, rewards, terminations, truncations, infos