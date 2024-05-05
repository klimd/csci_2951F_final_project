import bsuite
import gym
from bsuite.utils import gym_wrapper
from bsuite import sweep
import numpy as np

print('All possible values for bsuite_id:', sweep.SWEEP)
#@title Ids for an example experiment:
print('List bsuite_id for "bandit_noise" experiment:')
print(sweep.BANDIT_NOISE)

#@title List the configurations for the given experiment
for bsuite_id in sweep.BANDIT_NOISE:
  env = bsuite.load_from_id(bsuite_id)
  print('bsuite_id={}, settings={}, num_episodes={}'
        .format(bsuite_id, sweep.SETTINGS[bsuite_id], env.bsuite_num_episodes))

#@title Instantiate the environment corresponding to a given `bsuite_id`
env = bsuite.load_from_id("bandit_noise/0")


env = bsuite.load_and_record_to_csv('catch/0', results_dir='/path/to/results')
gym_env = gym_wrapper.GymFromDMEnv(env)

SAVE_PATH_RAND = '/tmp/bsuite/rand'
env = bsuite.load_and_record('bandit_noise/0', save_path=SAVE_PATH_RAND, overwrite=True)

for episode in range(env.bsuite_num_episodes):
  timestep = env.reset()
  while not timestep.last():
    action = np.random.choice(env.action_spec().num_values)
    timestep = env.step(action)