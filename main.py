import numpy as np
import gym
from physics_experiments.utils import get_dynamics_and_rewards, solve_unconstrained_v1
from physics_experiments.frozen_lake_env import ModifiedFrozenLake
from physics_experiments.visualization import plot_dist
from soft_q_learning import soft_q_learning
MAPS = {
    "2x9ridge": [
        "FFFFFFFFF",
        "FSFFFFFGF"
    ],
    "9x9zigzag": [
        "FFFFFFFFF",
        "FSFFFFFFF",
        "WWWWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFWWWWWW",
        "FFFFFFFGF",
        "FFFFFFFFF"
    ],
    "7x7zigzag": [
        "FFFFFFF",
        "FSFFFFF",
        "WWWWWFF",
        "FFFFFFF",
        "FFWWWWW",
        "FFFFFGF",
        "FFFFFFF"
    ]
}
def make_figure_3(max_beta=200, step=0.95, trajectory_length=10_000, eig_max_it=10_000_000, tolerance=1e-6):

    env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    data = []
    beta = max_beta / step
    while beta >= 1.:
        beta *= step
        print(f"beta={beta: 10.4f}", end=', ', flush=True)
        solution = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy, eig_max_it=eig_max_it, tolerance=tolerance)
        _, u, v, _, _, _ = solution

        dist = np.multiply(u, v.T)
        print(dist.shape)
        row = dict(
            beta=beta,
            solution=solution,
            distribution=dist,
        )
        data.append(row)

    dst_every = len(data) // 4
    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data) if i % dst_every == 0]

    title_list = []
    dist_list = []
    for beta, dist in reversed(dst_params[:-1]):
        dist = np.array(dist).reshape(nS, nA).sum(axis=1)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")
    plot_dist(env.desc, *dist_list, titles=title_list)

def soft_q_learning_figure_3(env, max_beta=200, step=0.95, trajectory_length=10_000, eig_max_it=10_000_000, tolerance=1e-6):

    agent = soft_q_learning(env)
    print(agent)
    data = []
    beta = max_beta / step
    while beta >= 1.:
        beta *= step
        print(f"beta={beta: 10.4f}", end=', ', flush=True)

        dist = agent.q_table
        row = dict(
            beta=beta,
            distribution=dist,
        )
        data.append(row)

    dst_every = len(data) // 4
    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data) if i % dst_every == 0]

    title_list = []
    dist_list = []
    for beta, dist in reversed(dst_params[:-1]):
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")
    plot_dist(env.desc, *dist_list, titles=title_list)

if __name__ == '__main__':

    desc = [    # 9x9zigzag
        "FFFFFFFFF",
        "FSFFFFFFF",
        "WWWWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFWWWWWW",
        "FFFFFFFGF",
        "FFFFFFFFF"
    ]
    # desc = ["SFF", "FFF", "HFG"]    #test-1
    env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)

    # make_figure_3(max_beta = 100, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)
    soft_q_learning_figure_3(env=env, max_beta = 100, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)





