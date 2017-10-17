import networkx
import numpy as np
from dataset import TRAIN, TEST
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from random import random


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
            -0.01, -0.02, -0.03, -0.04, -0.05, 0.0
        ]
        self.model = BayesianModel([
            ('Aware', 'AppDistraction'),
            ('Aware', 'CarDirection'),
            ('Aware', 'CarSound')
            # ('Aware', 'AwareTime')
        ])
        self.episodes = TEST.copy().drop('Aware', axis=1)

    def reset(self):
        self.model = BayesianModel([
            ('Aware', 'AppDistraction'),
            ('Aware', 'CarDirection'),
            ('Aware', 'CarSound')
            # ('Aware', 'AwareTime')
        ])
        # TODO: qual estimador usar? Por que? Isso deve estar no texto
        self.model.fit(TRAIN, estimator=MaximumLikelihoodEstimator)
        aware = [node for node in self.model.get_cpds() if node.variable == 'Aware'].pop()
        self.state = np.round(aware.values, 2)

        self.cpds = self._tabular_cpds_to_dict(self.model)

        return self.state

    def render(self):
        aware = [node for node in self.model.get_cpds() if node.variable == 'Aware'].pop()
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

        print('######## Episódio atual ########')
        print(episode)
        episode = {k: replacer[k][v] for k, v in episode.iteritems()}

        print('######## Estado atual - Aware ########')
        bp = BeliefPropagation(self.model)
        print(bp.query(['Aware'], evidence=episode)['Aware'])

        upper_bound = self.state[0] - adjustment
        lower_bound = self.state[1] + adjustment
        if adjustment == 0.0 and not (upper_bound > 1.0 or upper_bound < 0.0):
            adjustments = {
                'Aware': {
                    'Aware': upper_bound,
                    'Unaware': lower_bound
                }
            }

            state_values = [upper_bound, lower_bound]

            total_prob = self.total_probabilities(self.model, self.cpds, adjustments)

            for node in self.model.get_cpds():
                node.values = self._get_cpd_values(total_prob[node.variable])
        else:
            state_values = self.state.copy()

        print('É válido: ', self.model.check_model())
        print('######## Proximo estado - Aware ########')
        bp = BeliefPropagation(self.model)
        print(bp.query(['Aware'], evidence=episode)['Aware'])

        reward = float(input('Recompensa entre -1 e 1: '))
        next_state = np.round(state_values, 2)

        return next_state, reward

    def total_probabilities(self, model, tabular_cpds, changes=dict()):
        total_probability = {}

        leaves = model.get_leaves()
        root = list(model.get_roots()).pop()

        if changes.get(root):
            root_values = changes.get(root)

        for node, state in tabular_cpds.items():
            total_probability[node] = {}

            if node in leaves:
                for param, values in state.items():
                    # fixed
                    if changes.get(node) and changes.get(node).get(param):
                        total_prob = changes[node][param]
                    else:
                        total_prob = {
                            'Aware': values[0] * changes[root]['Aware'] + values[1] * changes[root]['Unaware'],
                            'Unaware': values[0] * changes[root]['Unaware'] + values[1] * changes[root]['Aware']
                        }

                    total_probability[node][param] = total_prob
            else:
                total_probability[node] = changes[node]

        return total_probability

