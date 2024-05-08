import numpy as np
import gym
from gym.wrappers import TimeLimit
from physics_experiments.frozen_lake_env import ModifiedFrozenLake

# This script allows us to get data on trajectory distributions and K-tables for K-Learning
def calculate_tau(sigma, beta_l, visitation_count):
     return (sigma ** 2 * beta_l)/(2 * visitation_count)

def k_learning(env, episodes, learning_rate, beta, sigma, discount_factor):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.full((nS, nA), 0)
    distributions = np.zeros((episodes, nS, nA))  # To store distributions after each episode
    q_table_at_each_t = np.zeros((episodes, nS, nA))
    visitation_count = np.zeros((nS, nA))
    for episode in range(episodes):
        beta_l =  beta * np.sqrt(episode + 1)
        state = env.reset()
        done = False
        truncated = False
        c = 0
        while not done and not truncated:
            c += 1
            ## For non-uniform sampling use the following p
            # p=np.exp(Q[state] / beta) / np.sum(np.exp(Q[state] / beta))
            action = np.random.choice(nA) 
            visitation_count[state][action] += 1
            next_state, reward, done, truncated, info = env.step(action)
            delta = (Q[next_state].min() + Q[next_state].max()) / 2
            tau = calculate_tau(sigma, beta_l, visitation_count[state, action])
            td_target = reward + tau + discount_factor * (delta + np.log(np.sum(np.exp(beta_l * (Q[next_state] - delta)) / nA)) / beta_l)
            if done:
                td_target = reward
            td_error = td_target - Q[state][action]
            Q[state][action] += learning_rate * td_error
            state = next_state
        q_table_at_each_t[episode] = Q
        # Compute distribution after the episode
        distributions[episode] = compute_state_distribution(env, Q, beta)

    return Q, distributions, q_table_at_each_t

def compute_state_distribution(env, Q, beta, num_simulations=100):
    state_count = np.zeros((env.observation_space.n, env.action_space.n))
    for _ in range(num_simulations):
        state = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action = np.random.choice(env.action_space.n, p=np.exp(beta * Q[state]) / np.sum(np.exp(beta * Q[state])))
            state_count[state, action] += 1
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
    
    # Smooth distribution
    state_count[state_count == 0] = 1
    return state_count / np.sum(state_count)

# Setting up the environment
env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)
env = TimeLimit(env, 250)

# Parameters
learning_rate = 0.1
beta = 20
sigma = 0.5
discount_factor = 1
episodes = 300

# Train using soft Q-learning and save distributions
Q, distributions, q_table_at_each_t = k_learning(env, episodes, learning_rate, beta, sigma, discount_factor)

np.save('k_tables.npy', q_table_at_each_t)
np.save('k_distributions.npy', distributions)



# Example usage: print distributions for the first and last episodes
print("Distribution after the first episode:", distributions[0])
print("Distribution after the last episode:", distributions[-1])
