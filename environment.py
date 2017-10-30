import networkx
import numpy as np
from dataset_analysis import TRAIN, TEST
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import BayesianEstimator


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

approximate = {
    'DistracaoApp': {
        'Nao': 0.00177777778,
        'Sim': -0.00177777778
    },
    'DirecaoCarro': {
        'Tras': -0.00111111111,
        'Direita': 0.00166666667,
        'Esquerda': 0.0022222222,
        'Frente': -0.0088888889
    },
    'SomCarro': {
        'Nao': 0.00155555556,
        'Sim': -0.00155555556
    },
    'Percepcao': {
        'Presente': -0.00622222222,
        'Tardia': -0.00377777778,
        'Ausente': 0.00997777778
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
        print('######## Episódio atual ########')
        print(episode)

        bp = BeliefPropagation(self.model)
        replaced_episode = {k: replacer[k][v] for k, v in episode.iteritems()}

        upper_bound = self.state[0] + adjustment
        lower_bound = self.state[1] - adjustment

        if not (upper_bound > 100 or lower_bound < 0):
            state_aware = [upper_bound, lower_bound]

            cpds = self._tabular_cpds_to_dict(self.model)
            adjustments = self.fit_probabilities(cpds, adjustment)
            for node in self.model.get_cpds():
                if node.variable != 'Consciente':
                    node.values = self._get_cpd_values(adjustments[node.variable])
                    node.normalize()
                else:
                    node.values = np.array(state_aware)

            for node in self.model.get_cpds():
                print(node)
        else:
            state_aware = [self.state]

        print('######## Proximo estado - Consciente ########')
        bp = BeliefPropagation(self.model)
        print(bp.query(['Consciente'], evidence=replaced_episode)['Consciente'])

        reward = float(input('Recompensa entre -1 e 1: '))
        next_state = []
        next_state.append(np.round(state_aware, 2))
        next_state.extend(list(replaced_episode.values()))

        return next_state, reward

    def fit_probabilities(self, cpds, adjustment):
        del cpds['Consciente']

        adjusted_probabilities = {}
        position = int(adjustment < 0)

        for state, param in cpds.items():
            params = list(param.keys())
            param_values = list(param.values())

            new_param_values = []
            npt = np.transpose(param_values)

            for cpd_list, param in zip(npt, params):
                fitting = approximate[state][param] * (adjustment * 100)

                values = []
                for cpd in cpd_list:
                    fit = cpd + fitting

                    if fit < 0:
                        fit = 0
                    elif fit > 1:
                        fit = 1

                    values.append(fit)

                new_param_values.append(self.normalize(values))

            npt = np.transpose(new_param_values)
            adjusted_probabilities[state] = {}

            for i, param in enumerate(params):
                adjusted_probabilities[state][param] = np.array(npt[i])

        return adjusted_probabilities

    def normalize(self, lst):
        s = sum(lst)
        return list(map(lambda x: float(x)/s, lst))

