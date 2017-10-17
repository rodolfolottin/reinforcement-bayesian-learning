import numpy as np
import random
from environment import AwareEnv, TEST
from collections import defaultdict
from pgmpy.inference import VariableElimination, BeliefPropagation


class Agent(object):
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0] * len(actions))

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        # using Bellman Optimality Equation to update q function
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = AwareEnv()
    agent = Agent(actions=list(range(len(env.actions))))

    for _ in range(50):
        state = env.reset()

        for episode in env.episodes.iterrows():
            env.render()

            index, episode = episode

            adjustment = agent.get_action(str(state))

            next_state, reward = env.step(env.actions[adjustment], episode)
            agent.learn(str(state), adjustment, reward, str(next_state))

            state = str(next_state)
            print(agent.q_table)

    print(state)

