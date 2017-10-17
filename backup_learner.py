import pandas as pd
import numpy as np
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, BdeuScore, K2Score, BicScore, ExhaustiveSearch, HillClimbSearch, ConstraintBasedEstimator
from pgmpy.models import BayesianModel
from pgmpy.independencies import Independencies

nodes = {
    'Aware',
    'AppDistraction',
    'CarSound',
    'CarDirection',
    'AwareTime'
}

header = ['id', 'car_id', 'Aware', 'AppDistraction', 'CarSound', 'CarDirection', 'AwareTime', '2', '3']
pandas_dataset = pd.read_csv('behaviour.csv', sep=';', names=header)
pandas_dataset.AppDistraction.replace(['Trivia', 'Headphone'], 'Yes', inplace=True)
pandas_dataset.AppDistraction.replace('Button', 'No', inplace=True)
# verificar diferença em retirar valores vazios ou nao
pandas_dataset = pandas_dataset.replace('-', np.nan)
# pandas_dataset = pandas_dataset.replace('-', np.nan).dropna()
pandas_dataset.drop(list(set(header) - nodes), axis=1, inplace=True)
# print(pandas_dataset['AppDistraction'].value_counts())

model = BayesianModel([('Aware', 'AppDistraction'), ('Aware', 'CarDirection'), ('Aware', 'CarSound')])  # Aware -> AppDistraction, Aware -> CarSound, Aware -> CarDirection
# estimador de maxima verossimilhança
model.fit(pandas_dataset, estimator=MaximumLikelihoodEstimator)

for value in model.get_cpds():
    print(value)

# model = BayesianModel([('Aware', 'AppDistraction'), ('Aware', 'CarDirection'), ('Aware', 'CarSound')])  # Aware -> AppDistraction, Aware -> CarSound, Aware -> CarDirection
 for cpd in model.get_cpds():
#     print(cpd)
# 

# structure learning
# scores para avaliar o quao bom sao dois modelos diferentes
bdeu = BdeuScore(pandas_dataset, equivalent_sample_size=5)
k2 = K2Score(pandas_dataset)
bic = BicScore(pandas_dataset)

model1 = BayesianModel([('Aware', 'AppDistraction'), ('Aware', 'CarDirection'), ('Aware', 'CarSound')])  # Aware -> AppDistraction, Aware -> CarSound, Aware -> CarDirection
model2 = BayesianModel([('AppDistraction', 'Aware'), ('CarDirection', 'Aware'), ('CarSound', 'Aware')])  # AppDistraction -> Aware, CarSound -> Aware, CarDirection -> Aware

print('modelo original')
print(bdeu.score(model1))
print(k2.score(model1))
print(bic.score(model1))

print('model invertido')
print(bdeu.score(model2))
print(k2.score(model2))
print(bic.score(model2))

# score entre locais, entre a ligacao de cada nodo
print(bdeu.local_score('AppDistraction', parents=[]))
print(bdeu.local_score('AppDistraction', parents=['Aware']))

# all dags
es = ExhaustiveSearch(pandas_dataset)
best_model = es.estimate()
print(best_model.edges())

print("\nAll DAGs by score:")
for score, dag in reversed(es.all_scores()):
    print(score, dag.edges())

# # use hill climb search to orient the edges:
# hc = HillClimbSearch(pandas_dataset, scoring_method=BicScore(pandas_dataset))
# model = hc.estimate()
# print("HillClimbSearch Model:    ", model.edges())
# 
# est = ConstraintBasedEstimator(pandas_dataset)
# print('ConstraintBasedEstimator')
# print(est.estimate(significance_level=0.01).edges())
 

# utiliza um metodo de sampling, existem outros ainda para criar uma amostra que seja parecida com a distribuição daquela
from pgmpy.models.BayesianModel import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

inference = BayesianModelSampling(model)

# sampling
model.fit(inference.forward_sample(10000), estimator=MaximumLikelihoodEstimator)

for value in model.get_cpds():
    print(value)


### Maneira de pegar cada um do dataset e ver qual seria o resultado desejado
### Esse fluxo aqui poderia jogar naquela nossa segunda opcao
### Um que nao entrasse ali eu faria ajustes automaticos, sei lá
aware_counter = defaultdict(int)
states_counter = defaultdict(int)
for index, row in predict_sample.iterrows():
    # TEMPO: row.to_dict() ja funcionaria porem os numeros vem como int64 do numpy
    evidence = {key: int(value) for key, value in row.iteritems()}

    predicted_value = infer.map_query([NODE], evidence=evidence)[NODE]

    # contagem
    aware_counter[inverse_replacer[NODE][predicted_value]] += 1

    if not sample.iloc[index][NODE] == predicted_value:

        states_counter[dumps(evidence)] += 1

for nodo, contagem in states_counter.items():
    nodo = {key: inverse_replacer[key][value] for key, value in loads(nodo).items()}

    print(nodo, contagem)

