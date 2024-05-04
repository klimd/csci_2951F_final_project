import gym
import numpy as np

class KLearning:
    def __init__(self, num_states, num_actions, alpha, gamma, beta, sigma):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.sigma = sigma
        self.k_table = np.zeros((num_states, num_actions))
        self.visitation_count = np.zeros((num_states, num_actions))

    def softmax(self, q_values):
        exp_values = np.exp(q_values / self.beta)
        return exp_values / np.sum(exp_values)

    def choose_action(self, state):
        k_values = self.k_table[state]
        action_probs = self.softmax(k_values)
        action = np.random.choice(self.num_actions, p=action_probs)
        return action
    
    def calculate_tau(self, beta_l, s, a):
        """Calculates the risk-seeking/termperature parameter from O'Donoghue"""
        return (self.sigma ** 2 * beta_l)/(2 * self.visitation_count[s, a])

    def update_k_table(self, beta_l, state, action, reward, next_state):
        k_current = self.k_table[state, action]
        k_next = self.k_table[next_state]
        v_next = np.log(np.sum(np.exp(k_next * self.beta))) / self.beta
        tau = self.calculate_tau(beta_l, state, action)
        k_target = reward + tau + self.gamma * v_next # TODO possibly change reward to mean reward
        self.k_table[state, action] += self.alpha * (k_target - k_current)

    def train(self, num_episodes, env):
        for episode in range(1, num_episodes + 1):
            beta_l = self.beta * np.sqrt(episode)
            state, info = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                self.visitation_count[state, action] += 1 
                next_state, reward, done, truncated, info = env.step(action)
                self.update_k_table(beta_l, state, action, reward, next_state)
                state = next_state

    def evaluate(self, env, num_episodes):
        total_rewards = []
        for episode in range(1, num_episodes + 1):
            state, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = np.argmax(self.k_table[state])
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')

    # Some Hyperparameters
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    alpha = 0.1
    gamma = 0.99
    beta = 0.1
    sigma = 1

    # Training Agent
    agent = KLearning(num_states, num_actions, alpha, gamma, beta, sigma)
    num_episodes = 10000
    agent.train(num_episodes, env)

    # Evaluation
    num_eval_episodes = 100
    avg_reward = agent.evaluate(env, num_eval_episodes)
    print(f"Average reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.k_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        state = next_state