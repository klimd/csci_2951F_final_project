from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from numpy.random import SeedSequence, default_rng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from scipy.sparse import lil_matrix

from frozen_lake_env import ModifiedFrozenLake
from utils import get_dynamics_and_rewards, solve_unconstrained, solve_unconstrained_v1, get_mdp_transition_matrix, test_policy
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

def make_figure_2():
    N = 250
    env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)
    env = TimeLimit(env, N)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    beta = 20
    solution = solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-6)
    l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solution
    bulk_dist = np.array(np.multiply(u, v.T))

    # The MDP transition matrix
    P = get_mdp_transition_matrix(dynamics, prior_policy)

    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()

    # The twisted matrix (and its transpose)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()

    # initialize left eigenvector
    u = np.matrix(np.ones((nSnA, 1)))

    # initialize right eigenvector
    # we might consider using the initial distribution here to fix initial conditions
    # v = np.multiply(np.matrix(env.isd).T, prior_policy).flatten().T
    # but in general we choose a uniform distribution
    v = np.matrix(np.ones((nSnA, 1)))

    # we will be keeping track of the forward and backward messages at each time step
    u_in_time = np.zeros((N + 1, nSnA), dtype=float)
    v_in_time = np.zeros((N + 1, nSnA), dtype=float)

    u_in_time[N] = u.T
    v_in_time[0] = v.T

    for i in range(1, N + 1):
        uk = (Mt).dot(u)
        lu = np.sum(uk)
        uk = uk / lu

        vk = M.dot(v)
        lv = np.sum(vk)
        vk = vk / lv

        # update the eigenvectors
        u = uk
        v = vk
        u_in_time[N - i] = u.T
        v_in_time[i] = v.T

    dist_in_time = u_in_time * v_in_time
    dist_in_time /= dist_in_time.sum(axis=1).reshape((-1, 1))

    # # use the bulk distribution as reference
    # pt = dist_in_time
    # q = bulk_dist.flatten()
    # qt = np.broadcast_to(q, pt.shape)

    # use the true time dependent distribution as the reference
    qt = dist_in_time
    p = bulk_dist.flatten()
    pt = np.broadcast_to(p, qt.shape)

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
    plt.legend(loc='upper center', prop={'size': 12})
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
        solution = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy, eig_max_it=eig_max_it,
                                          tolerance=tolerance)
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
        dist = np.array(dist).reshape(nS, nA).sum(axis=1)
        dist_list.append(dist)
        title_list.append(rf"$\beta$ = {beta:.2f}")
    plot_dist(env.desc, *dist_list, titles=title_list)

    N = trajectory_length
    ncpu = cpu_count()
    n_episodes = ncpu * 2
    rngs = [default_rng(s) for s in SeedSequence().spawn(ncpu)]

    def work(policy, rng):
        env = ModifiedFrozenLake(map_name='9x9zigzag', min_reward=-2.)
        env = TimeLimit(env, N)
        return [-test_policy(env, policy, quiet=True, rng=rng) / N for _ in range(n_episodes // ncpu)]

    print('Running simulation ...')
    sim = np.array(
        [(beta, np.array(sum(Parallel(n_jobs=ncpu)(delayed(work)(policy, rng) for rng in rngs), [])).mean())
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


def solve_dp(beta, max_steps, gamma):
    env = ModifiedFrozenLake(map_name='10x10empty')
    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    q_table = np.matrix(np.zeros((nS, nA)))
    v_vectr = np.matrix(np.zeros((nS, 1)))

    for _ in range(max_steps):
        q_table = (rewards + gamma * dynamics.multiply(v_vectr).sum(axis=0)).reshape((nS, nA))

        # delta variable helps with numerical stability
        delta = (q_table.min(axis=1) + q_table.max(axis=1)) / 2

        v_vectr = delta + np.log(np.multiply(np.exp(beta * (q_table - delta)), prior_policy).sum(axis=1)) / beta

    return q_table


def make_figure_5(beta=10, max_steps=300):
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
    print(ld_q_table_t)
    print(ld_q_table_t.shape)
    ld_q_table_t[1:] /= t[1:]

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
    y = ld_q_table_t[t].A.flatten()
    x = q_table_t[t].A.flatten()
    ax = fig.axes[2]
    ax.scatter(x, y, label=f'Q values for N = {t}')
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    ax.set_xlabel('DP Soft-Q values / N')
    ax.set_ylabel('Large Deviation Soft-Q values / N')
    ax.legend()

    t = 290
    y = ld_q_table_t[t].A.flatten()
    x = q_table_t[t].A.flatten()
    ax = fig.axes[3]
    ax.scatter(x, y, label=f'Q values for N = {t}')
    ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--')
    ax.set_xlabel('DP Soft-Q values / N')
    ax.set_ylabel('Large Deviation Soft-Q values / N')
    ax.legend()

    plt.tight_layout()
    plt.show()


def make_figure_6(beta=20):
    for map_name in ['7x7holes', '8x8zigzag', '9x9ridgex4']:
        env = ModifiedFrozenLake(map_name=map_name, min_reward=-1.5)

        dynamics, rewards = get_dynamics_and_rewards(env)
        nS, nSnA = dynamics.shape
        nA = nSnA // nS
        prior_policy = np.ones((nS, nA)) / nA

        solution = solve_unconstrained_v1(beta, dynamics, rewards, prior_policy, eig_max_it=100_000, tolerance=5e-6)
        l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solution
        plot_dist(env.desc, optimal_policy)


def make_figure_7(beta_list=[1, 5, 10, 50, 100], alpha=0.01, episode_length=1_000, n_episodes=2_000, n_epochs=5):
    N = episode_length
    env = ModifiedFrozenLake(map_name='7x7holes')
    env = TimeLimit(env, N)
    l_alpha = alpha / env.nS / env.nA

    dynamics, rewards = get_dynamics_and_rewards(env)
    prior_policy = np.ones((env.nS, env.nA)) / env.nA

    print("Collecting experience ...")
    replay_memory = []

    pbar = tqdm(range(n_episodes), ncols=120)
    for i in pbar:
        # keep SARSA structure
        state = env.reset()
        done = truncated = False
        action = np.random.choice(env.nA, p=prior_policy[state])
        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = np.random.choice(env.nA, p=prior_policy[next_state])
            replay_memory.append((state, action, reward, next_state, next_action))
            state, action = next_state, next_action

    n_training_steps_available = len(replay_memory)

    print("\nBegin training process ...")
    for beta in beta_list:
        print()

        # initialize
        l_value = 1.
        u_table = np.matrix(np.ones((env.nS, env.nA)))
        u_scale = np.linalg.norm(u_table.A.flatten())

        rw_sim_list = [np.mean([test_policy(env, prior_policy) for _ in range(10)]) / N]
        rw_sim_step_list = [0]

        evaluation_period = N * 100
        rescale_period = N * 100

        steps_trained = 0
        for epoch in range(1, n_epochs + 1):
            pbar = tqdm(range(n_training_steps_available), ncols=120)
            for step in pbar:
                state, action, reward, next_state, next_action = replay_memory[step]
                exp_re = np.exp(reward * beta)

                u_valu = u_table[state, action]
                u_next = u_table[next_state, next_action]
                with np.errstate(divide='ignore', over='ignore'):
                    ratio = u_next / u_valu

                if ratio != np.inf:
                    u_valu = (1. - alpha) * u_valu + alpha * exp_re / l_value * u_next
                    u_table[state, action] = u_valu
                    l_value = (1. - l_alpha) * l_value + l_alpha * exp_re * ratio
                    l_value = min(l_value, 1)

                steps_trained += 1

                if step % (evaluation_period) == 0:
                    policy = np.multiply(u_table, prior_policy)
                    policy /= policy.sum(axis=1)

                    evaluation = np.mean([test_policy(env, policy.A) for _ in range(10)]) / N
                    rw_sim_list.append(evaluation)
                    rw_sim_step_list.append(steps_trained)

                    theta = -np.log(l_value) / beta
                    pbar.set_description(
                        f'Beta: {beta: 5.1f}. Epoch: {epoch}/{n_epochs}. Evaluation: {evaluation: 8.4f}. Theta={theta: 5.1e}')

                if step % (rescale_period) == 0:
                    # helps with numerical stability
                    u_table /= np.linalg.norm(u_table.A.flatten()) * u_scale

        plt.plot(rw_sim_step_list, rw_sim_list, label=f"beta: {beta}")

    plt.xlabel('Training step')
    plt.ylabel('Mean per step reward')
    plt.legend()
    plt.show()
    print()


def make_figure_8(beta=10, n_replicas=3, n_episodes=2_000):
    N = 1_000

    env = ModifiedFrozenLake(map_name='7x7holes')
    env = TimeLimit(env, N)

    dynamics, rewards = get_dynamics_and_rewards(env)
    nS, nSnA = dynamics.shape
    nA = nSnA // nS
    prior_policy = np.ones((nS, nA)) / nA

    def work(replica, rng):
        show_prg = replica == 1
        if show_prg:
            print("\nShowing progress for replica #1. The rest is running in parallel ...")

        env = ModifiedFrozenLake(map_name='7x7holes')
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
        if show_prg:
            print("Collecting experience ...")
            pbar = tqdm(pbar, ncols=120)
        for _ in pbar:
            # keep SARSA structure
            state = env.reset()
            done = truncated = False
            action = rng.choice(env.nA, p=prior_policy[state])
            while not (done or truncated):
                state_freq[state] += 1
                next_state, reward, done, truncated, _ = env.step(action)
                next_action = rng.choice(env.nA, p=prior_policy[next_state])
                exp_re = np.exp(reward * beta)

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

                state, action = next_state, next_action

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


if __name__ == '__main__':
    # print('\nMaking figure 2 ... ')
    # make_figure_2()
    #
    # print('\nMaking figure 3 ... ')
    # make_figure_3(max_beta=100, step=0.80, trajectory_length=5_000, eig_max_it=1_000_000, tolerance=5e-4)

    print('\nMaking figure 5 ... ')
    make_figure_5(beta=10, max_steps=300)
    #
    # print('\nMaking figure 6 ... ')
    # make_figure_6(beta=20)
    #
    # print('\nMaking figure 7 (faster version). This should take about 2 minutes ... ')
    # make_figure_7(beta_list=[5, 10], alpha=0.02, episode_length=1_000, n_episodes=1_000, n_epochs=2)
    #
    # # print('\nMaking figure 7 (fast version). This should take about 5 minutes ... ')
    # # make_figure_7(beta_list = [5, 10, 50], alpha = 0.01, episode_length = 1_000, n_episodes = 1_000, n_epochs=6)
    #
    # # print('\nMaking figure 7. This should take about 10 minutes ... ')
    # # make_figure_7(beta_list = [1, 5, 10, 50, 100], alpha = 0.01, episode_length = 1_000, n_episodes = 2_000, n_epochs = 5)
    #
    # print('\nMaking figure 8 (faster version). This should take about 1 minute ... ')
    # make_figure_8(beta=10, n_replicas=2, n_episodes=1_000)
    #
    # # print('\nMaking figure 8 (fast version). This should take about 10 minutes ... ')
    # # make_figure_8(beta = 10, n_replicas = 2, n_episodes = 10_000)
    #
    # # print('\nMaking figure 8. This should take about 100 minutes ... ')
    # # make_figure_8(beta = 10, n_replicas = 2, n_episodes = 100_000)