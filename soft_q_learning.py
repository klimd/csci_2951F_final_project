import gym
import numpy as np

class SoftQLearning:
    def __init__(self, num_states, num_actions, alpha, gamma, beta):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.q_table = np.zeros((num_states, num_actions))

    def softmax(self, q_values):
        exp_values = np.exp(q_values / self.beta)
        return exp_values / np.sum(exp_values)

    def choose_action(self, state):
        q_values = self.q_table[state]
        action_probs = self.softmax(q_values)
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def update_q_table(self, state, action, reward, next_state):
        q_current = self.q_table[state, action]
        q_next = self.q_table[next_state]
        v_next = np.log(np.sum(np.exp(q_next * self.beta))) / self.beta
        q_target = reward + self.gamma * v_next # TODO possibly change reward to mean reward
        self.q_table[state, action] += self.alpha * (q_target - q_current)

    def train(self, num_episodes, env):
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def evaluate(self, env, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = np.argmax(self.q_table[state])
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)
    
# if __name__ == '__main__':
def soft_q_learning(env):
    # env = gym.make('FrozenLake-v1', desc=["SFF", "FFF", "HFG"], is_slippery=True)

    # Some Hyperparameters
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    alpha = 0.1
    gamma = 0.99
    beta = 100

    # Training Agent
    agent = SoftQLearning(num_states, num_actions, alpha, gamma, beta)
    num_episodes = 100
    agent.train(num_episodes, env)

    # Evaluation
    num_eval_episodes = 100
    avg_reward = agent.evaluate(env, num_eval_episodes)
    print(f"Average reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, truncated, info = env.step(action)
        print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
        state = next_state

    return agent