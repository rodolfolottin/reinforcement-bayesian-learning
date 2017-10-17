import pandas as pd
import numpy as np
import networkx
from json import loads, dumps
from collections import defaultdict
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.independencies import Independencies
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from sklearn.model_selection import train_test_split


nodes = {
    'Aware',
    'AppDistraction',
    'CarSound',
    'CarDirection',
    'AwareTime'
}

header = [
    'id',
    'car_id',
    'Aware',
    'AppDistraction',
    'CarSound',
    'CarDirection',
    'AwareTime',
    'Questions',
    'CorrectQuestions'
]

dataset = pd.read_csv('behaviour.csv', sep=';', names=header)
dataset.AppDistraction.replace(['Trivia', 'Headphone'], 'Yes', inplace=True)
dataset.AppDistraction.replace('Button', 'No', inplace=True)
dataset = dataset.replace('-', np.nan).dropna()
dataset.drop(list(set(header) - nodes), axis=1, inplace=True)
train, test = train_test_split(dataset, test_size=0.2)


class AwareEnv(object):
    def __init__(self):
        self.actions = [
            (0, 0),
            (0.05, -0.05),
            (0.10, -0.10),
            (-0.05, 0.05),
            (-0.10, 0.10)
        ]
        self.bayesian_model = BayesianModel([
            ('Aware', 'AppDistraction'),
            ('Aware', 'CarDirection'),
            ('Aware', 'CarSound')
            # ('Aware', 'AwareTime')
        ])
        # TODO verificar qual estimador usar
        self.bayesian_model.fit(train, estimator=MaximumLikelihoodEstimator)
        self.episodes = test.copy().drop('Aware', axis=1)

    def reset(self):
        self.bayesian_model.fit(train, estimator=MaximumLikelihoodEstimator)
        self.episodes = test.copy().drop('Aware', axis=1)

        last_episode = self.episodes.iloc[0]
        self.episodes = self.episodes.iloc[0:]
        return last_episode

    def step(self, action):

        infer = VariableElimination(self.bayesian_model)
        predict = infer.map_query(['Aware'], evidence=episode)['Aware']

        # if test.loc[index]['Aware'] != action:
        #     print(test.loc[index])
        #     print(self.episodes.loc[index])

        #     print(infer.query(['Aware'], evidence))

        #     input()

        # index, episode = episode

        # episode = {k: replacer[k][v] for k, v in episode.iteritems()}

        # infer = VariableElimination(state)


        # TODO verificar se por um caso o self.episodes dá pop pelo iloc
        if len(self.episodes) == 0:
            done = True
            next_state = None
        else:
            next_state = self.episodes.iloc[0]
            self.episodes = self.episodes.iloc[1:]
            done = False


        return done, next_state
        # return inverse_replacer['Aware'][action]

