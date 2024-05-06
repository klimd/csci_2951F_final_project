import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

def get_transitions_and_rewards(env):

    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    transitions = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    colsums = transitions.sum(axis=0)
    assert (colsums.round(12) == 1.).all(), f"{colsums.min()=}, {colsums.max()=}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)), shape=shape).sum(axis=0)

    return transitions, rewards

def solve_unconstrained_optimization(beta, transitions, rewards, prior_policy, eig_max_iterations=10000, tolerance=1e-8):
    scale = 1 / np.exp(beta * rewards.min())

    num_states, num_states_times_num_actions = transitions.shape
    num_actions = num_states_times_num_actions // num_states

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(transitions, prior_policy)

    # Diagonal of exponentiated rewards (multiplied by beta)
    # This is a structure for constructing sparse matrices incrementally
    T = lil_matrix((num_states_times_num_actions, num_states_times_num_actions))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()

    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()

    # left eigenvector
    u = np.matrix(np.ones((num_states_times_num_actions, 1))) * scale
    u_scale = np.linalg.norm(u)

    # right eigenvector
    v = np.matrix(np.ones((num_states_times_num_actions, 1))) * scale
    v_scale = np.linalg.norm(v)

    lol = float('inf')
    hil = 0.

    metrics_list = []

    for i in range(1, eig_max_iterations + 1):

        uk = (Mt).dot(u)
        lu = np.linalg.norm(uk) / u_scale
        uk = uk / lu

        vk = M.dot(v)
        lv = np.linalg.norm(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100_000 == 0:
            metrics_list.append(dict(
                lu=lu,
                lv=lv,
                u_err=u_err,
                v_err=v_err,
            ))

        if u_err <= tolerance and v_err <= tolerance:
            l = lu
            print(f"iter={i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu
        print(
            f"Did not converge: {i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")

    l = lu

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((num_states, num_actions)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((num_states, num_actions)), prior_policy).sum(axis=1)
    X = transitions.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(u, v.T).reshape((num_states, num_actions)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def get_dynamics_and_rewards(env):
    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    colsums = dynamics.sum(axis=0)
    assert (colsums.round(12) == 1.).all(), f"{colsums.min()=}, {colsums.max()=}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)), shape=shape).sum(axis=0)

    return dynamics, rewards


def solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):
    tolerance *= beta

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()
    M_scale = 1.

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1)))
    u_scale = np.sum(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * nSnA ** 2
    v_scale = np.sum(v)

    lol = float('inf')
    hil = 0.

    for i in range(1, eig_max_it + 1):

        uk = (Mt).dot(u)
        lu = np.sum(uk) / u_scale
        mask = np.logical_and(uk > 0., uk < np.inf)
        rescale = 1. / np.sqrt(uk[mask].max() * uk[mask].min())
        uk = uk / lu * rescale
        u_scale *= rescale

        vk = M.dot(v)
        lv = np.sum(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100 == 0:
            rescale = 1 / np.sqrt(lu)
            Mt = Mt * rescale
            M_scale *= rescale

        if u_err <= tolerance and v_err <= tolerance:
            # if u_err <= tolerance:
            l = lu / M_scale
            print(
                f"{i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {M_scale=:.4e}, {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu / M_scale
        print(
            f"Did not converge: {i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {M_scale=:.4e}, {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")

    l = lu / M_scale

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_unconstrained_v1(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):
    scale = 1 / np.exp(beta * rewards.min())

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1))) * scale
    u_scale = np.linalg.norm(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * scale
    v_scale = np.linalg.norm(v)

    lol = float('inf')
    hil = 0.

    metrics_list = []

    for i in range(1, eig_max_it + 1):

        uk = (Mt).dot(u)
        lu = np.linalg.norm(uk) / u_scale
        uk = uk / lu

        vk = M.dot(v)
        lv = np.linalg.norm(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100_000 == 0:
            metrics_list.append(dict(
                lu=lu,
                lv=lv,
                u_err=u_err,
                v_err=v_err,
            ))

        if u_err <= tolerance and v_err <= tolerance:
            l = lu
            print(f"iter={i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu
        print(
            f"Did not converge: {i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")

    l = lu

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def compute_max_kl_divergence(dist_a, dist_b, axis=0):
    numer = csr_matrix(dist_a)
    denom = coo_matrix(dist_b)
    kldiv = denom.copy()
    numer = np.array(numer[denom.row, denom.col]).flatten()
    kldiv.data = np.log(numer / denom.data) * numer
    kldiv = kldiv.sum(axis=axis)

    return kldiv.max()


def compute_policy_induced_distribution(dynamics, policy, steps, isd):
    mdp_generator = get_mdp_transition_matrix(dynamics, policy)

    x = np.multiply(np.matrix(isd).T, policy).flatten().T
    for _ in range(steps):
        x = mdp_generator.dot(x)

    return x.T


def get_mdp_transition_matrix(transition_dynamics, policy):
    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA + np.array(list(range(nA)))).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)), (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows, cols)), shape=(nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix


def test_policy(env, policy, quiet=True, rng=None):
    if rng is not None:
        random_choice = rng.choice
    else:
        random_choice = np.random.choice

    state = env.reset()

    done = truncated = False
    episode_reward = 0
    while not (done or truncated):
        # Sample action from action probability distribution
        action = random_choice(env.action_space.n, p=policy[state])

        # Apply the sampled action in our environment
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

    if not quiet:
        print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' ' * 10 + '\n', flush=True)

    return episode_reward