import numpy as np
import random
from environment import AwareEnv
from collections import defaultdict
from pgmpy.inference import VariableElimination, BeliefPropagation


replacer = {
    'AppDistraction': {
        'No': 0,
        'Yes': 1
    },
    'CarDirection': {
        'Back': 0,
        'Front': 1,
        'Left': 2,
        'Right': 3
    },
    'Aware': {
        'Aware': 0,
        'Unaware': 1
    },
    'CarSound': {
        'Off': 0,
        'On': 1
    },
    'AwareTime': {

    }
}

inverse_replacer = {
    outer_k: {
        inner_v: inner_k for inner_k, inner_v in outer_v.items()
    }
    for outer_k, outer_v in replacer.items()
}


class Agent(object):
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            print(state.to_dict())
            input()
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    # update q function with sample <s, a, r>
    def learn(self, state, action, reward):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor
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
    # vamos supor que as ações sejam subir um valor de um nodo qualquer ou nao
    agent = Agent(actions=list(env.actions))

    for episode in range(50):
        state = env.reset()

        while True:
            # de acordo com o que no estado o agente tomaria a ação?
            # meu estado do ambiente é um episodio em específico!
            # entao como base naquelas características vai ser salvo no dicionário
            # a pontuação que aquele nodo teria p possíveis ajustes na rede bayesiana!
            action = agent.get_action(state)

            # com base nas ações eu vou ao step e verifico
            # o quão Aware realmente ele estaria aos olhos do especialista?
            # aí que entra a minha função de avaliação
            # e com base nisso eu posso apenas alterar o nodo Aware
            # isso alterará para os outros nodos também!

            # next_state conforme os jogos são alterações possíveis
            # que o agente toma no ambiente
            # e como o ambiente responde a eles...
            # nao consigo enxergar aqui o que que seria
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward)

            state = next_state

            # if episode ends, then break
            if done:
                break
