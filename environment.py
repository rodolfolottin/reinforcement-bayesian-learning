import networkx
import numpy as np
from dataset_analysis import TRAIN, TEST
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from random import random


replacer = {
    'DistracaoApp': {
        'Nao': 0,
        'Sim': 1
    },
    'DirecaoCarro': {
        'Tras': 3,
        'Direita': 0,
        'Esquerda': 1,
        'Frente': 2
    },
    'Consciente': {
        'Consciente': 0,
        'Inconsciente': 1
    },
    'SomCarro': {
        'Nao': 0,
        'Sim': 1
    },
    'Percepcao': {
        'Presente': 1,
        'Tardia': 2,
        'Ausente': 0
    }
}

inverse_replacer = {
    outer_k: {
        inner_v: inner_k for inner_k, inner_v in outer_v.items()
    }
    for outer_k, outer_v in replacer.items()
}


class AwareEnv(object):
    def __init__(self):
        self.actions = [
            0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
            -0.01, -0.02, -0.03, -0.04, -0.05
        ]
        self.model = BayesianModel([
            ('Consciente', 'DistracaoApp'),
            ('Consciente', 'DirecaoCarro'),
            ('Consciente', 'SomCarro'),
            ('Consciente', 'Percepcao')
        ])
        self.episodes = TEST.copy().drop('Consciente', axis=1)

    def reset(self):
        self.model = BayesianModel([
            ('Consciente', 'DistracaoApp'),
            ('Consciente', 'DirecaoCarro'),
            ('Consciente', 'SomCarro'),
            ('Consciente', 'Percepcao')
        ])
        self.model.fit(TRAIN, estimator=BayesianEstimator)
        aware = [node for node in self.model.get_cpds() if node.variable == 'Consciente'].pop()
        self.state = [np.round(aware.values, 2)]

        self.cpds = self._tabular_cpds_to_dict(self.model)

        for node in self.model.get_cpds():
            print(node)
        input()

        return self.state

    def render(self):
        aware = [node for node in self.model.get_cpds() if node.variable == 'Consciente'].pop()
        self.state = np.round(aware.values, 2)

        self.cpds = self._tabular_cpds_to_dict(self.model)

    def _tabular_cpds_to_dict(self, model):
        return {
            node.variable: {
                state: value for state, value in zip(node.state_names[node.variable], node.values)
            }
            for node in model.get_cpds()
        }

    def _get_cpd_values(self, node_values):
        cpds = []

        for state, param in node_values.items():
            if type(param) == dict:
                cpds.append(list(param.values()))
            else:
                cpds.append(param)

        return np.array(cpds)

    def step(self, adjustment, episode):
        print('######## Ajustes ########')
        print(adjustment)

        upper_bound = self.state[0] - adjustment
        lower_bound = self.state[1] + adjustment

        if adjustment != 0.0 and not (upper_bound > 1.0 or upper_bound < 0.0):

        print('######## Episódio atual ########')
        print(episode)

        episode = {k: replacer[k][v] for k, v in episode.iteritems()}

        upper_bound = self.state[0] - adjustment
        lower_bound = self.state[1] + adjustment

        print('######## Proximo estado - Consciente ########')
        bp = BeliefPropagation(self.model)
        print(bp.query(['Consciente'], evidence=episode)['Consciente'])

        reward = float(input('Recompensa entre -1 e 1: '))

            state_aware = [upper_bound, lower_bound]
            adjustments = {
                'Consciente': {
                    'Consciente': upper_bound,
                    'Inconsciente': lower_bound
                }
            }

            cpds = self._tabular_cpds_to_dict(self.model)
            adjustments = self.adjust_probabilities(self.model, cpds, adjustments)
            for node in self.model.get_cpds():
                if node.variable != 'Consciente':
                    new_cpds = self._get_cpd_values(adjustments[node.variable])
                    node.values = node.values / new_cpds
                    node.normalize()
        else:
            state_aware = [self.state.copy()]

        next_state = []
        next_state.append(np.round(state_aware, 2))
        next_state.extend(list(episode.values()))

        return next_state, reward

    # def adjust_probabilities(self, model, tabular_cpds, changes=dict()):
    #     adjusted_probability = {}

    #     leaves = model.get_leaves()
    #     root = list(model.get_roots()).pop()

    #     for node, state in tabular_cpds.items():
    #         adjusted_probability[node] = {}

    #         if node in leaves:
    #             for param, values in state.items():
    #                 prob = {}
    #                 prob['Consciente'] = values[0] * changes[root]['Consciente'] + values[1] * changes[root]['Inconsciente']
    #                 prob['Inconsciente'] = values[0] * changes[root]['Inconsciente'] + values[1] * changes[root]['Consciente']

    #                 adjusted_probability[node][param] = prob
    #         else:
    #             adjusted_probability[node] = changes[node]

    #     return adjusted_probability

    def adjust_probabilities(self, model, changes, episode):
        tabular_cpds = self._tabular_cpds_to_dict(model)
        adjusted_probability = {}

        leaves = model.get_leaves()
        root = list(model.get_roots()).pop()

        print(tabular_cpds)
        for node, state in tabular_cpds.items():
            adjusted_probability[node] = {}

            if node in leaves:
                for param, values in state.items():

                    adjusted_probability[node][param] = prob
            else:
                adjusted_probability[node] = changes[node]

        return adjusted_probability

