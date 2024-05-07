import numpy as np
import matplotlib.pyplot as plt
import gym

from gym.wrappers import TimeLimit
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from numpy.random import SeedSequence, default_rng

from k_learning import k_learning
from physics_experiments.utils import get_dynamics_and_rewards, solve_unconstrained_v1, solve_unconstrained
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

def make_figure_2_soft_q(n = 250):
    N = n
    env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)
    env = TimeLimit(env, N)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    beta = 10
    solution = solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-6)
    l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solution
    bulk_dist = np.array(np.multiply(u, v.T))

    new_desc = [    # 9x9zigzag
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

    excluded_wall_indices = []
    for i, c in enumerate(''.join(new_desc)):
        if c == 'W':
            for j in range(4):
                excluded_wall_indices.append(i * 4 + j)

    # env = gym.make('FrozenLake-v1', desc=desc, is_slippery=True)

    # agent = soft_q_learning(env, beta=beta, num_episodes=N, eval=True)
    dist_in_time = np.load('distributions.npy')#agent.state_action_dist_t
    q_in_time = np.load('q_tables.npy')#agent.state_action_dist_t
    #dist_in_time = np.exp(dist_in_time / beta)
    dist_in_time /= dist_in_time.sum(axis=(1, 2), keepdims=True)
    dist_in_time = dist_in_time.reshape(N, -1)

    # Mask out walls
    dist_in_time = np.delete(dist_in_time, excluded_wall_indices, axis=1)
    # use the true time dependent distribution as the reference
    pt = dist_in_time
    q = bulk_dist.flatten()
    q = np.delete(q, excluded_wall_indices)
    qt = np.broadcast_to(q, pt.shape)

    mt = pt > 0
    test = all([(p[m] > 0).all() and (q[m] > 0).all() for p, m, q in zip(pt, mt, qt)])
    assert test, "Error, zero q elements found in p(x) * log (p(x)/q(x)). Dkl would be invalid"

    with np.errstate(divide='ignore', invalid='ignore'):
        logpt = np.log(pt)
        logqt = np.log(qt)
        x = pt * (logpt - logqt)
    x[np.isnan(x)] = 0.

    Dkl_t = x.sum(axis=1)
    fig = plt.figure(figsize=(5, 2), dpi=150)
    fig.subplots_adjust(bottom=0.2, left=0.07, right=0.99, top=0.95)
    plt.plot(Dkl_t, label=r'$D_{kl}(t)$')
    plt.hlines(0., 0 - 5, N + 5, 'k', '--')
    plt.xlabel(r'$t$')
    plt.ylim(bottom=-2)
    plt.legend(loc='lower right', prop={'size': 12})
    plt.show()
    plt.close()

    
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
    for beta in [2, 20, 40, 200]:
    # for beta in [200, 150, 100, 50]:
        print(f"beta={beta}", end=', ', flush=True)

        agent = soft_q_learning(env, beta=beta)

        dist = agent.state_distribution
        row = dict(
            beta=beta,
            distribution=dist,
        )
        data.append(row)

    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data)]

    title_list = []
    dist_list = []
    for beta, dist in dst_params:
        print(dist)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")

    plot_dist(env.desc, *dist_list, titles=title_list)

def k_learning_figure_3(env, max_beta=200, step=0.95, trajectory_length=10_000, eig_max_it=10_000_000, tolerance=1e-6):

    data = []
    # for beta in [2, 20, 40, 200]:
    # for beta in [0.5, 1, 1.5, 2]:
    # for beta in [0.5, 0.05, 0.025, 0.005 ]:
    for beta in [50, 100, 200]:
        print(f"beta={beta}", end=', ', flush=True)

        agent = k_learning(env, beta=beta)

        dist = agent.state_distribution
        row = dict(
            beta=beta,
            distribution=dist,
        )
        data.append(row)

    dst_params = [(r['beta'], r['distribution']) for i, r in enumerate(data)]

    title_list = []
    dist_list = []
    for beta, dist in dst_params:
        # dist = np.array(dist).reshape(agent.num_states, agent.num_actions).sum(axis=1)
        print(dist)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")

    plot_dist(env.desc, *dist_list, titles=title_list)

def soft_q_learning_figure_5(beta=200, max_steps=300):

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

    
    N = max_steps
    # new_env = gym.make('FrozenLake-v1', desc=new_desc, is_slippery=True)
    # new_env._max_episode_steps = N
    env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2)
    env = TimeLimit(env, N)
    # We want to exclude values that are in a wall by masking them out because they are just our initial values
    excluded_wall_indices = []
    for i, c in enumerate(''.join(new_desc)):
        if c == 'W':
            for j in range(4):
                excluded_wall_indices.append(i * 4 + j)

    # agent = k_learning(env, beta, N) #soft_q_learning(env=env, beta=beta, num_episodes=max_steps)
    # print(agent.q_table_at_each_t.shape)
    #q_table_t = agent.q_table_at_each_t[:agent.t_final, :, :]
    # q_table_t = agent.q_table_at_each_t[:N, :, :]
    q_table_t = np.load('q_tables.npy')
    # q_table_t = q_table_t.reshape(N, agent.num_states * agent.num_actions)
    q_table_t = q_table_t.reshape(max_steps, env.observation_space.n * env.action_space.n)
    print(q_table_t.shape)


    #env = ModifiedFrozenLake(map_name='10x10empty', min_reward=-2.)
    

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA
    solution = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy)
    l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solution

    # t = np.matrix(np.arange(agent.t_final)).T
    t = np.matrix(np.arange(N)).T
    # here we create a q_table from eigenvalue and left-eigenvector, for each trajectory length
    # this is to compare directly with the ground truth q_table from DP
    #ld_q_table_t = (-l * t + np.log(u)) / beta
    #ld_q_table_t[1:] /= t[1:]
    ld_q_table_t = np.full((1, nSnA), l) + np.log(u) / beta
    ld_q_table_t = np.asmatrix(np.full((N, nSnA), ld_q_table_t))
    
    
    print(ld_q_table_t.shape)

    fig = plot_dist(env.desc, env.desc, None, None, None, show_plot=False, ncols=2)

    # Normalize
    slice_min = q_table_t.min(axis=(1), keepdims=True)
    slice_max = q_table_t.max(axis=(1), keepdims=True)
    normalized_qtt = (q_table_t - slice_min) / (slice_max - slice_min)

    slice_min = ld_q_table_t.A.min(axis=(1), keepdims=True)
    slice_max = ld_q_table_t.A.max(axis=(1), keepdims=True)
    normalized_ldqtt = (ld_q_table_t - slice_min) / (slice_max - slice_min)
    
    # Mask out wall indices
    normalized_qtt = np.delete(normalized_qtt, excluded_wall_indices, axis=1)
    normalized_ldqtt = np.delete(normalized_ldqtt, excluded_wall_indices, axis=1)

    s, d = 1, 10
    y = np.sqrt(np.power(normalized_qtt - normalized_ldqtt, 2).mean(axis=1)).A.flatten()[s::d]
    x = t.A.flatten()[s::d]
    ax = fig.axes[1]
    ax.scatter(x, y)
    ax.set_yscale('log')
    ax.set_ylabel('Soft-Q values /N RMSE')
    ax.set_xlabel('Episode Length (steps)')

    t = 20
    y = normalized_ldqtt[t].A.flatten()
    # x = q_table_t[t].A.flatten()
    x = normalized_qtt[t].flatten()
    ax = fig.axes[2]
    ax.scatter(x, y, label=f'Q values for N = {t}')
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    ax.set_xlabel('Learned Soft-Q values / N')
    ax.set_ylabel('Large Deviation Soft-Q values / N')
    ax.legend()

    t = N - 1 
    y = normalized_ldqtt[t].A.flatten()
    # y = y - y.min()
    # y = y / y.max()
    # x = q_table_t[t].A.flatten()
    x = normalized_qtt[t].flatten()
    # mn = x.min()
    # x = x - mn
    # x = x / x.max()
    ax = fig.axes[3]
    ax.scatter(x, y, label=f'Q values for N = {t + 1}')
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    ax.set_xlabel('Learned Soft-Q values / N')
    ax.set_ylabel('Large Deviation Soft-Q values / N')
    ax.legend()

    plt.tight_layout()
    plt.show()

def make_figure_8_with_k_learning(beta=10, n_replicas=3, n_episodes=2_000):
    N = 1_000

    env = ModifiedFrozenLake(map_name='7x7holes')
    # env = ModifiedFrozenLake(map_name='9x9zigzag')
    env = TimeLimit(env, N)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    sigma = 0.5
    def work(replica, rng):
        show_prg = replica == 1
        if show_prg:
            print("\nShowing progress for replica #1. The rest is running in parallel ...")

        env = ModifiedFrozenLake(map_name='7x7holes')
        # env = ModifiedFrozenLake(map_name='9x9zigzag')
        env = TimeLimit(env, N)
        prior_policy = np.ones((nS, nA)) / nA

        state_freq = np.zeros(env.nS) + np.finfo(float).eps

        # initialize
        l_value = np.exp(-beta)
        u_table = np.matrix(np.ones((env.nS, env.nA)))
        v_table = np.matrix(np.ones((env.nS, env.nA)))

        save_period = N * n_episodes // 1000
        rescale_period = save_period * 100

        # simplistic choice for initial learning rate that scales with
        init_alpha = np.log10(n_episodes) / n_episodes * 30
        alpha = init_alpha
        l_alpha = init_alpha / env.nS

        theta_list = [-np.log(l_value) / beta]
        step_list = [0]
        alpha_list = [alpha]
        l_alpha_list = [l_alpha]

        steps_trained = 0

        pbar = range(n_episodes)
        l = 1
        visitation_count = np.matrix(np.ones((env.nS, env.nA)))

        if show_prg:
            print("Collecting experience ...")
            pbar = tqdm(pbar, ncols=120)
        for _ in pbar:
            # keep SARSA structure
            state = env.reset()
            done = truncated = False
            action = rng.choice(env.nA, p=prior_policy[state])

            visitation_count_before_l = visitation_count

            while not (done or truncated):
                state_freq[state] += 1
                next_state, reward, done, truncated, _ = env.step(action)
                next_action = rng.choice(env.nA, p=prior_policy[next_state])
                # exp_re = np.exp(reward * beta)

                visitation_count[state, action] += 1
                exp_re = np.exp(beta * (reward + (sigma ** 2 * np.sqrt(l)) / (2 * visitation_count_before_l[state, action])))

                ##### the right eigenvector update #####
                v_valu = v_table[next_state, next_action]
                v_prev = v_table[state, action]
                bayes_rule = prior_policy[next_state, next_action] * state_freq[next_state] / (
                            prior_policy[state, action] * state_freq[state])
                v_valu = v_valu + alpha * (exp_re / l_value * v_prev * bayes_rule - v_valu)
                v_table[next_state, next_action] = v_valu

                ##### the left eigenvector update #####
                u_valu = u_table[state, action]
                u_next = u_table[next_state, next_action]
                u_valu = u_valu + alpha * (exp_re / l_value * u_next - u_valu)
                u_table[state, action] = u_valu

                ##### the eigenvalue update #####
                l_value = l_value + l_alpha * (exp_re * u_next / u_valu - l_value)
                l_value = min(l_value, 1)

                steps_trained += 1

                if steps_trained % (save_period) == 0:
                    theta = -np.log(l_value) / beta
                    theta_list.append(theta)

                    step_list.append(steps_trained)
                    alpha_list.append(alpha)
                    l_alpha_list.append(l_alpha)

                if steps_trained % (rescale_period) == 0:
                    v_table /= v_table.sum()
                    u_table /= np.multiply(u_table, v_table).sum()

                    alpha = alpha * 0.55
                    l_alpha = l_alpha * 0.55

                    # alpha = alpha * 0.25
                    # l_alpha = l_alpha * 0.25

                state, action = next_state, next_action

            l += 1
        return dict(
            step_list=step_list,
            theta_list=theta_list,
            alpha_list=alpha_list,
            l_alpha_list=l_alpha_list,
        )

    ncpu = cpu_count()
    if n_replicas > ncpu:
        print(f"Will use one replica for each available CPU. n_replicas is now = {ncpu}")
        n_replicas = ncpu
    rngs = [default_rng(s) for s in SeedSequence().spawn(n_replicas)]
    data = Parallel(n_jobs=n_replicas)(delayed(work)(i + 1, rng) for i, rng in enumerate(rngs))

    plt.figure(figsize=(12, 4))
    ax = plt.subplot2grid((1, 2), (0, 1))
    x = data[0]['step_list']
    y1 = data[0]['alpha_list']
    y2 = data[0]['l_alpha_list']
    ax.plot(x, y1, color='C2', label="$\\alpha$")
    ax.plot(x, y2, color='C3', label="$\\alpha_\\theta$")
    ax.set_xlabel('Training step')
    ax.set_yscale('log')
    ax.set_ylabel("Learning rate")
    ax.legend()

    ax = plt.subplot2grid((1, 2), (0, 0))
    gt_eigenvalue = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy)[0]
    gt_theta = -np.log(gt_eigenvalue) / beta
    x, y = [], []
    for d in data:
        x.append(d['step_list'])
        y.append(d['theta_list'])
    x = np.array(x).mean(axis=0)
    y = np.array(y)
    e = np.abs(y - gt_theta)[:, -1].max()
    y, dy = y.mean(axis=0), y.std(axis=0)
    ax.plot(x, y)
    ax.fill_between(x, y - dy, y + dy, alpha=0.4)
    ax.hlines(gt_theta, x[0], x[-1], linestyles='dashed', color='black', label=r'Target $\\theta$')
    ax.set_xlabel('Training step')
    ax.set_ylim(gt_theta - e * 10, gt_theta + e * 10)
    ax.set_ylabel("Learned $\\theta$ value\n(mean over replicas)")

    print(f"True theta:  {gt_theta}")
    print(f"Theta error: {e}  (max over {n_replicas} replicas)")
    print(f"Theta relative error: {e / gt_theta * 100:.4f}%  (max over {n_replicas} replicas)")
    plt.tight_layout()
    plt.show()

def make_figure_new():
    n_replicas = 2
    n_episodes = 1_00
    betas = [50, 100, 150, 200]
    sigmas = [0.25, 0.5, 0.75, 1]

    N = 1_000

    env = ModifiedFrozenLake(map_name='7x7holes')
    # env = ModifiedFrozenLake(map_name='9x9zigzag')
    env = TimeLimit(env, N)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    for beta in betas:
        for sigma in sigmas:
            def work(replica, rng):
                show_prg = replica == 1
                if show_prg:
                    print("\nShowing progress for replica #1. The rest is running in parallel ...")

                env = ModifiedFrozenLake(map_name='7x7holes')
                # env = ModifiedFrozenLake(map_name='9x9zigzag')
                env = TimeLimit(env, N)
                prior_policy = np.ones((nS, nA)) / nA

                state_freq = np.zeros(env.nS) + np.finfo(float).eps

                # initialize
                l_value = np.exp(-beta)
                u_table = np.matrix(np.ones((env.nS, env.nA)))
                v_table = np.matrix(np.ones((env.nS, env.nA)))

                save_period = N * n_episodes // 1000
                rescale_period = save_period * 100

                # simplistic choice for initial learning rate that scales with
                init_alpha = np.log10(n_episodes) / n_episodes * 30
                alpha = init_alpha
                l_alpha = init_alpha / env.nS

                theta_list = [-np.log(l_value) / beta]
                step_list = [0]
                alpha_list = [alpha]
                l_alpha_list = [l_alpha]

                steps_trained = 0

                pbar = range(n_episodes)
                l = 1
                visitation_count = np.matrix(np.ones((env.nS, env.nA)))

                if show_prg:
                    print("Collecting experience ...")
                    pbar = tqdm(pbar, ncols=120)
                for _ in pbar:
                    # keep SARSA structure
                    state = env.reset()
                    done = truncated = False
                    action = rng.choice(env.nA, p=prior_policy[state])

                    visitation_count_before_l = visitation_count

                    while not (done or truncated):
                        state_freq[state] += 1
                        next_state, reward, done, truncated, _ = env.step(action)
                        next_action = rng.choice(env.nA, p=prior_policy[next_state])
                        # exp_re = np.exp(reward * beta)

                        visitation_count[state, action] += 1
                        exp_re = np.exp(beta * (reward + (sigma ** 2 * np.sqrt(l)) / (2 * visitation_count_before_l[state, action])))

                        ##### the right eigenvector update #####
                        v_valu = v_table[next_state, next_action]
                        v_prev = v_table[state, action]
                        bayes_rule = prior_policy[next_state, next_action] * state_freq[next_state] / (
                                    prior_policy[state, action] * state_freq[state])
                        v_valu = v_valu + alpha * (exp_re / l_value * v_prev * bayes_rule - v_valu)
                        v_table[next_state, next_action] = v_valu

                        ##### the left eigenvector update #####
                        u_valu = u_table[state, action]
                        u_next = u_table[next_state, next_action]
                        u_valu = u_valu + alpha * (exp_re / l_value * u_next - u_valu)
                        u_table[state, action] = u_valu

                        ##### the eigenvalue update #####
                        l_value = l_value + l_alpha * (exp_re * u_next / u_valu - l_value)
                        l_value = min(l_value, 1)

                        steps_trained += 1

                        if steps_trained % (save_period) == 0:
                            theta = -np.log(l_value) / beta
                            theta_list.append(theta)

                            step_list.append(steps_trained)
                            alpha_list.append(alpha)
                            l_alpha_list.append(l_alpha)

                        if steps_trained % (rescale_period) == 0:
                            v_table /= v_table.sum()
                            u_table /= np.multiply(u_table, v_table).sum()

                            alpha = alpha * 0.55
                            l_alpha = l_alpha * 0.55

                            # alpha = alpha * 0.25
                            # l_alpha = l_alpha * 0.25

                        state, action = next_state, next_action

                    l += 1
                return dict(
                    step_list=step_list,
                    theta_list=theta_list,
                    alpha_list=alpha_list,
                    l_alpha_list=l_alpha_list,
                )

            ncpu = cpu_count()
            if n_replicas > ncpu:
                print(f"Will use one replica for each available CPU. n_replicas is now = {ncpu}")
                n_replicas = ncpu
            rngs = [default_rng(s) for s in SeedSequence().spawn(n_replicas)]
            data = Parallel(n_jobs=n_replicas)(delayed(work)(i + 1, rng) for i, rng in enumerate(rngs))

        ax = plt.subplot2grid((1, 2), (0, 0))
        gt_eigenvalue = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy)[0]
        gt_theta = -np.log(gt_eigenvalue) / beta
        x, y = [], []
        for d in data:
            x.append(d['step_list'])
            y.append(d['theta_list'])
        x = np.array(x).mean(axis=0)
        y = np.array(y)
        e = np.abs(y - gt_theta)[:, -1].max()
        y, dy = y.mean(axis=0), y.std(axis=0)
        ax.plot(x, y)
        ax.fill_between(x, y - dy, y + dy, alpha=0.4)
        ax.hlines(gt_theta, x[0], x[-1], linestyles='dashed', color='black', label=r'Target $\\theta$')
        ax.set_xlabel('Training step')
        ax.set_ylim(gt_theta - e * 10, gt_theta + e * 10)
        ax.set_ylabel("Learned $\\theta$ value\n(mean over replicas)")

        print(f"True theta:  {gt_theta}")
        print(f"Theta error: {e}  (max over {n_replicas} replicas)")
        print(f"Theta relative error: {e / gt_theta * 100:.4f}%  (max over {n_replicas} replicas)")
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
    # soft_q_learning_figure_3(env=env, max_beta = 200, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)
    # k_learning_figure_3(env=env, max_beta = 200, step = 0.80, trajectory_length = 5_000, eig_max_it=10_000_000,  tolerance = 5e-4)
    make_figure_2_soft_q(300)
    soft_q_learning_figure_5(beta = 10, max_steps=300)
    

    # # print('\nMaking figure 8 (fast version). This should take about 10 minutes ... ')
    # make_figure_8_with_k_learning(beta = 10, n_replicas = 2, n_episodes = 10_000)