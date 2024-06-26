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
        self.t_final = 0

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
        q_target = reward + self.gamma * v_next 
        self.q_table[state, action] += self.alpha * (q_target - q_current)
       

    def train(self, num_episodes, env, eval = False):
        self.q_table_at_each_t = np.zeros((num_episodes, self.num_states, self.num_actions))
        self.state_action_dist_t = np.zeros((num_episodes, self.num_states, self.num_actions))
        for episode in range(num_episodes):
            state = env.reset()
            if type(state) is tuple:
                state, _ = state 
            done = False
            truncated = False
            while not done and not truncated:
                action = np.random.choice(self.num_actions, p=np.exp(self.q_table[state] / self.beta) / np.sum(np.exp(self.q_table[state] / self.beta)))
                next_state, reward, done, truncated, info = env.step(action)
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                if done:
                    td_target = reward
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.alpha * td_error
                state = next_state
                self.q_table_at_each_t[episode] = self.q_table
            if eval:
                self.evaluate(env, 100)
                self.state_action_dist_t[episode] = self.state_action_distribution




    def evaluate(self, env, num_episodes):
        total_rewards = []
        state_action_distribution = np.zeros((self.num_states, self.num_actions))
        for episode in range(num_episodes):
            state = env.reset()
            if type(state) is tuple:
                state, _ = state 
            done = False
            truncated = False
            episode_reward = 0
            
            while not done and not truncated:
                action = self.choose_action(state)
                state_action_distribution[state, action] += 1
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        self.state_action_distribution = state_action_distribution
        return np.mean(total_rewards)
    
# if __name__ == '__main__':
def soft_q_learning(env, beta=100, num_episodes=100, eval= False):

    # Some Hyperparameters
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    alpha = 0.1
    gamma = 1
    beta = beta

    # Training Agent
    agent = SoftQLearning(num_states, num_actions, alpha, gamma, beta)
    agent.train(num_episodes, env)

    # Evaluation
    num_eval_episodes = 100
    avg_reward = agent.evaluate(env, num_eval_episodes)
    print(f"Average reward over {num_eval_episodes} episodes: {avg_reward:.2f}")
    # state, info = env.reset()
    # done = False
    # while not done:
    #     action = np.argmax(agent.q_table[state])
    #     next_state, reward, done, truncated, info = env.step(action)
    #     # print(f"State: {state}, Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    #     state = next_state

    return agent