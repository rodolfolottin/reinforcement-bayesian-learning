import numpy as np
import random
import json
import os
from environment import AwareEnv
from collections import defaultdict


class Agent(object):
    def __init__(self, actions, q_table):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0] * len(actions), q_table)

    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
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
    q_table = {}

    if os.stat('q_table.json').st_size > 0:
        answer = input('Foi encontrado um arquivo q_table contendo uma tabela já preenchida. Gostaria de continuar utilizando-na? Y/N: ')

        with open('q_table.json', 'r+') as qtable_json:
            if answer.upper() == 'Y':
                q_table = json.load(qtable_json)
            else:
                qtable_json.seek(0)
                qtable_json.truncate()

    env = AwareEnv()
    agent = Agent(actions=list(range(len(env.actions))), q_table=q_table)

    try:
        for _ in range(10):
            state = env.reset()

            for episode in env.episodes.iterrows():
                env.render()

                index, episode = episode

                adjustment = agent.get_action(str(state))

                next_state, reward = env.step(env.actions[adjustment], episode)
                agent.learn(str(state), adjustment, reward, str(next_state))

                state = str(next_state)
    finally:
        print('\n ######## Rede Bayesiana Final ########')

        for node in env.model.get_cpds():
            print(node)

        with open('q_table.json', 'w') as qtable_json:
            json.dump(agent.q_table, qtable_json, indent=4)

