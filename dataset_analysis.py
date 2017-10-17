import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.independencies import Independencies
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, BdeuScore, K2Score, BicScore, ExhaustiveSearch, HillClimbSearch, ConstraintBasedEstimator


nodes = {
    'Aware',
    'AppDistraction',
    'CarSound',
    'CarDirection',
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


dataset = pd.read_csv('behaviour.csv', sep=';', names=header)
dataset.AppDistraction.replace(['Trivia', 'Headphone'], 'Yes', inplace=True)
dataset.AppDistraction.replace('Button', 'No', inplace=True)
dataset = dataset.replace('-', np.nan).dropna()
dataset.drop(list(set(header) - nodes), axis=1, inplace=True)
SPLIT = int(len(dataset) * 0.7)
TRAIN = dataset[:SPLIT].copy()
TEST = dataset[SPLIT:].copy()


def discretize_awaretime(dataset):
    for ran in (4, 6, 8, 10, 12, 14, 16, 18, 20):
        for function in (pd.qcut, pd.cut):
            ser, bins = function(dataset.AwareTime, ran, labels=False, retbins=True)

            plt.title("Histograma AwareTime {} Bins".format(str(ran)))
            plt.xlabel("Tempo")
            plt.ylabel("Tamanho")

            plt.hist(dataset.AwareTime, bins=bins, edgecolor="k")
            plt.xticks(bins)

            if function == pd.cut:
                text = 'Intervalos iguais'
            else:
                text = 'Frequenciais iguais'

            plt.savefig('Figura {}-{}'.format(str(ran), text))
            plt.close()


def best_dag(dataset):
    # all dags
    es = ExhaustiveSearch(dataset)
    best_model = es.estimate()
    print(best_model.edges())

    print("\nAll DAGs by score:")


def apply_k_means():
    pass

# discretize_awaretime(dataset)
# best_dag(dataset)

