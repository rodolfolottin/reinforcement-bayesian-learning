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
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, BdeuScore, K2Score, BicScore, ExhaustiveSearch, HillClimbSearch, ConstraintBasedEstimator
from sklearn.cluster import KMeans

nodes = {
    'Aware',
    'AppDistraction'
    # 'CarSound',
    # 'CarDirection'
    # 'AwareTime'
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

## Normalização e discretização dos dados
## Leio CSV configurando o header e como se separam os valores
## Dou replace no conjunto de dados em Trivia e Headphone para que há intereferência do Aplicativo e Button que é o caso em que não há
## Separo valores desconhecidos para um valor que a rede poderá receber
pandas_dataset = pd.read_csv('behaviour.csv', sep=';', names=header)
pandas_dataset.AppDistraction.replace(['Trivia', 'Headphone'], 'Yes', inplace=True)
pandas_dataset.AppDistraction.replace('Button', 'No', inplace=True)
pandas_dataset = pandas_dataset.replace('-', np.nan).dropna()
pandas_dataset.drop(list(set(header) - nodes), axis=1, inplace=True)

## Crio meu modelo bayesiano
# Aware -> AppDistraction, Aware -> CarSound, Aware -> CarDirection

# model = BayesianModel([('AppDistraction', 'Aware'), ('CarDirection', 'Aware'), ('CarSound', 'Aware')])
model = BayesianModel([('Aware', 'AppDistraction')])

## Estimo o quanto ele se adaptaria
# Verificar quais das estimações eu vou utilizar como meu modelo
SPLIT = int(len(pandas_dataset) * 0.7)

train_dataset = pandas_dataset[:SPLIT].copy()
test_dataset = pandas_dataset[SPLIT:].copy()

model.fit(pandas_dataset, estimator=MaximumLikelihoodEstimator)
# bayesian estimator
# model.fit(pandas_dataset, estimator=BayesianEstimator, prior_type="BDeu")

tabular_cpds = {
    node.variable: {
        state: value for state, value in zip(node.state_names[node.variable], node.values)
    }
    for node in model.get_cpds()
}

def total_probabilities(model, tabular_cpds, changes=dict()):
    """
    Recalculates the total probabilities given the expert feedback
    P(Node = State) = P(Node = State ⋂  Aware) U P(Node = State ⋂  NOT Aware)

    Parameters
    ----------
    model: BayesianModel instance
        BayesianModel to be used

    changes: dict
        A dict containing the new values of given variables

    Returns
    -------
    dict: Returns a dict containing the total probabilities of the bayesian model
    """
    total_probability = {}

    leaves = model.get_leaves()
    root = list(model.get_roots()).pop()

    if changes.get(root):
        root_values = changes.get(root)
    # elif changes:
    #     root_states = list(tabular_cpds[root].keys())
    #     node, state = list(changes.items()).pop()
    #     state, value = list(state.items()).pop()

    #     prob, not_prob = tabular_cpds[node][state]

    #     changes[root] = {
    #         root_states[1]: root_value,
    #         root_states[0]: 1 - root_value
    #     }

    #     return total_probabilities(model, tabular_cpds, changes=changes)

    for node, state in tabular_cpds.items():
        total_probability[node] = {}

        if node in leaves:
            for param, values in state.items():
                # fixed
                if changes.get(node) and changes.get(node).get(param):
                    total_prob = changes[node][param]
                else:
                    total_prob = values[0] * changes[root]['Aware'] + values[1] * changes[root]['Unaware']

                total_probability[node][param] = total_prob
        else:
            total_probability[node] = changes[node]

    return total_probability


lista = [value.to_factor() for value in model.get_cpds()]

print(lista[0])
print(lista[1])

root = lista[1].copy()
root.values[0] -= 0.4
root.values[1] += 0.4

print(root)
result = lista[0] * lista[1] / root
result.normalize()

print(result)

