import numpy as np
import matplotlib.pyplot as plt
import gym

from gym.wrappers import TimeLimit
from k_learning import k_learning
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
        print(dist)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")

    plot_dist(env.desc, *dist_list, titles=title_list)

def soft_q_learning_figure_3(env, max_beta=200, step=0.95, trajectory_length=10_000, eig_max_it=10_000_000, tolerance=1e-6):

    data = []
    # for beta in [2, 20, 40, 200]:
    for beta in [200, 150, 100, 50]:
        print(f"beta={beta}", end=', ', flush=True)

        agent = soft_q_learning(env, beta=beta)

        dist = agent.state_action_distribution
        row = dict(
            beta=beta,
            distribution=dist,
        )
        data.append(row)

    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data)]

    title_list = []
    dist_list = []
    for beta, dist in reversed(dst_params):
        dist = np.array(dist).reshape(agent.num_states, agent.num_actions).sum(axis=1)
        print(dist)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")

    plot_dist(env.desc, *dist_list, titles=title_list)

def k_learning_figure_3(env, max_beta=200, step=0.95, trajectory_length=10_000, eig_max_it=10_000_000, tolerance=1e-6):

    data = []
    # for beta in [2, 20, 40, 200]:
    for beta in [0.5, 1, 1.5, 2]:
        print(f"beta={beta}", end=', ', flush=True)

        agent = k_learning(env, beta=beta)

        dist = agent.k_table
        row = dict(
            beta=beta,
            distribution=dist,
        )
        data.append(row)

    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data)]

    title_list = []
    dist_list = []
    for beta, dist in reversed(dst_params):
        dist = np.array(dist).reshape(agent.num_states, agent.num_actions).sum(axis=1)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")

    plot_dist(env.desc, *dist_list, titles=title_list)

def soft_q_learning_figure_5(beta=10, max_steps=300):
    N = max_steps

    # env = ModifiedFrozenLake(map_name='10x10empty', min_reward=-2.)
    env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)
    env = TimeLimit(env, N)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    q_table_t = np.matrix(np.zeros((N + 1, nSnA)))
    v_vectr_t = np.matrix(np.zeros((N + 1, nS)))

    for steps_left in range(1, N + 1):
        v_vectr_next = v_vectr_t[steps_left - 1].T
        q_table = (rewards + dynamics.multiply(v_vectr_next).sum(axis=0)).reshape((nS, nA))
        # delta variable helps with numerical stability
        delta = (q_table.min(axis=1) + q_table.max(axis=1)) / 2
        v_vectr = delta + np.log(np.multiply(np.exp(beta * (q_table - delta)), prior_policy).sum(axis=1)) / beta

        q_table_t[steps_left] = q_table.flatten()
        v_vectr_t[steps_left] = v_vectr.flatten()

    solution = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy)
    l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solution

    t = np.matrix(np.arange(len(q_table_t))).T
    q_table_t[1:] /= t[1:]

    # here we create a q_table from eigenvalue and left-eigenvector, for each trajectory length
    # this is to compare directly with the ground truth q_table from DP
    ld_q_table_t = (np.log(l) * t + np.log(u)) / beta
    # print(ld_q_table_t)
    print(ld_q_table_t.shape)
    ld_q_table_t[1:] /= t[1:]

    new_desc = [  # 9x9zigzag
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
    new_env = gym.make('FrozenLake-v1', desc=new_desc, is_slippery=True)
    agent = soft_q_learning(env=new_env, beta=200)
    print(agent.q_table.shape)
    ld_q_table_t = agent.q_table.reshape(1, -1)
    # print(ld_q_table_t)
    print(ld_q_table_t.shape)
    # Tile the reshaped array to create a (301, 324) array
    ld_q_table_t = np.tile(ld_q_table_t, (301, 1))
    print(ld_q_table_t.shape)


    fig = plot_dist(env.desc, env.desc, None, None, None, show_plot=False, ncols=2)

    s, d = 1, 10
    y = np.sqrt(np.power(q_table_t - ld_q_table_t, 2).mean(axis=1)).A.flatten()[s::d]
    x = t.A.flatten()[s::d]
    ax = fig.axes[1]
    ax.scatter(x, y)
    ax.set_yscale('log')
    ax.set_ylabel('Soft-Q values /N RMSE')
    ax.set_xlabel('Episode Length (steps)')

    t = 20
    # y = ld_q_table_t[t].A.flatten()
    y = ld_q_table_t[t]
    x = q_table_t[t].A.flatten()
    ax = fig.axes[2]
    ax.scatter(x, y, label=f'Q values for N = {t}')
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    ax.set_xlabel('DP Soft-Q values / N')
    ax.set_ylabel('Large Deviation Soft-Q values / N')
    ax.legend()

    t = 290
    # y = ld_q_table_t[t].A.flatten()
    y = ld_q_table_t[t]
    x = q_table_t[t].A.flatten()
    ax = fig.axes[3]
    ax.scatter(x, y, label=f'Q values for N = {t}')
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    ax.set_xlabel('DP Soft-Q values / N')
    ax.set_ylabel('Large Deviation Soft-Q values / N')
    ax.legend()

    plt.tight_layout()
    plt.show()


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

    # make_figure_3(max_beta = 2, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)
    soft_q_learning_figure_3(env=env, max_beta = 200, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)
    # k_learning_figure_3(env=env, max_beta = 200, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)

    # soft_q_learning_figure_5(beta=10, max_steps=300)




