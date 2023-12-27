from pettingzoo.test import parallel_api_test
from extended_pd import parallel_env


env = parallel_env()
parallel_api_test(env, num_cycles=1000)