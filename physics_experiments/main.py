from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from numpy.random import SeedSequence, default_rng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from scipy.sparse import lil_matrix

from frozen_lake_env import ModifiedFrozenLake
from utils import get_transitions_and_rewards, solve_unconstrained_optimization, get_mdp_transition_matrix, test_policy
from visualization import plot_dist

def steady_state_distribution_for_various_betas(max_beta=200, step=0.95, trajectory_length=10_000, eig_max_iterations=10_000_000, tolerance=1e-6):

    env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)

    dynamics, rewards = get_transitions_and_rewards(env)
    num_states, num_states_times_num_actions = dynamics.shape
    num_actions = num_states_times_num_actions // num_states
    prior_policy = np.ones((num_states, num_actions)) / num_actions

    data = []
    beta = max_beta / step
    while beta >= 1.:
        beta *= step
        print(f"beta={beta: 10.4f}", end=', ', flush=True)
        solution = solve_unconstrained_optimization(beta, dynamics, rewards, prior_policy, eig_max_iterations=eig_max_iterations, tolerance=tolerance)
        l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solution

        dist = np.multiply(u, v.T)
        row = dict(
            beta=beta,
            E=- np.multiply(dist, rewards).sum(),
            S1=np.multiply(dist, -np.log(optimal_policy.flatten())).sum(),
            S2=np.multiply(dist, np.log(prior_policy.flatten())).sum(),
            F=-np.log(l) / beta,
            l=l,
            theta=-np.log(l),
            solution=solution,
            distribution=dist,
        )
        data.append(row)

    sim_every = len(data) // 10
    sim_params = [(r['beta'], r['solution'][3]) for i, r in enumerate(data) if i % sim_every == 0]
    dst_every = len(data) // 4
    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data) if i % dst_every == 0]
    for d in data:
        del (d['solution'], d['distribution'])

    title_list = []
    dist_list = []
    for beta, dist in reversed(dst_params[:-1]):
        dist = np.array(dist).reshape(num_states, num_actions).sum(axis=1)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")
    plot_dist(env.desc, *dist_list, titles=title_list)

    N = trajectory_length
    ncpu = cpu_count()
    n_episodes = ncpu * 2
    ranges = [default_rng(s) for s in SeedSequence().spawn(ncpu)]

    def work(policy, rng):
        env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)
        env = TimeLimit(env, N)
        return [-test_policy(env, policy, quiet=True, range=rng) / N for _ in range(n_episodes//ncpu)]

    print('Running simulation ...')
    sim = np.array(
        [(beta, np.array(sum(Parallel(n_jobs=ncpu)(delayed(work)(policy, rng) for rng in ranges), [])).mean())
         for beta, policy in sim_params])
    print('Done')

    df = pd.DataFrame(data)
    df = df.sort_values('beta')

    _, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 4))
    ax = axes[0]
    ax.plot(df.beta, df.F, label='Free Energy')
    ax.plot(df.beta, df.E, label='Energy')
    ax.plot(*sim.T, 'o', label='Simulated Energy    ')
    ax.set_ylabel('Mean cost per step')
    ax.legend(loc='lower left')

    ax = axes[1]
    ax.plot(df.beta, -(df.S1 + df.S2), label='Relative Entropy    ')
    ax.set_ylabel('Mean entropy per step')
    ax.legend()

    plt.xlabel('beta')
    plt.xscale('log')
    plt.xlim(1, df.beta.max())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    print('\nMaking figure 3 ... ')
    steady_state_distribution_for_various_betas(max_beta = 100, step = 0.80, trajectory_length = 5_000, eig_max_iterations = 1_000_000, tolerance = 5e-4)
