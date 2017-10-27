import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pgmpy.models import BayesianModel


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

dataset = pd.read_csv('datasets/behaviour.csv', sep=';', names=header)
dataset.drop(list(set(header) - nodes), axis=1, inplace=True)
# TRADUÇÃO
dataset.rename(
    columns={
        'Aware': 'Consciente',
        'AppDistraction': 'DistracaoApp',
        'CarSound': 'SomCarro',
        'CarDirection': 'DirecaoCarro',
        'AwareTime': 'Percepcao'
    }, inplace=True
)

dataset = dataset.replace('-', np.nan).dropna()
dataset.DistracaoApp.replace(['Trivia', 'Headphone'], 'Sim', inplace=True)
dataset.DistracaoApp.replace('Button', 'Nao', inplace=True)
dataset.Consciente.replace(['Aware', 'Unaware'], ['Consciente', 'Inconsciente'], inplace=True)
dataset.SomCarro.replace(['On', 'Off'], ['Sim', 'Nao'], inplace=True)
dataset.DirecaoCarro.replace(['Back', 'Front', 'Left', 'Right'], ['Tras', 'Frente', 'Esquerda', 'Direita'], inplace=True)


def get_mean_aware_value_by_direction_soundtype(dataset):
    mean_aware_by_type = {}

    df = dataset
    for direction in ('Direita', 'Esquerda', 'Tras', 'Frente'):
        mean_aware_by_type[direction] = {}

        for sound in ('Sim', 'Nao'):
            df_split = df[(df.DirecaoCarro == direction) & (df.SomCarro == sound)]
            mean_aware_by_type[direction][sound] = df_split.Percepcao.mean()
            # max_aware_by_type[direction][sound] = df_split.loc[df_split['Percepcao'].mean()].Percepcao

    return mean_aware_by_type


def generate_line_plot(mean_aware):
    sound_on = []
    sound_off = []

    for car_dir, sound in mean_aware.items():
        for sound_type, mean_value in sound.items():
            if sound_type == 'Sim':
                sound_on.append(mean_value)
            else:
                sound_off.append(mean_value)

    # Data
    # df=pd.DataFrame({'Sim': sound_on, 'Não': sound_off})
    # df.cumsum()
    # ax = df.plot(linewidth=2, marker='')

    # ax.set_xticks(df.index)
    # ax.set_xticklabels(['Direita', 'Esquerda', 'Trás', 'Frente'], rotation=0)
    # # Histograma utilizado
    # plt.legend(title='SomCarro')
    # plt.title("Média de tempo para Consciente")
    # plt.ylabel("Tempo")
    # plt.savefig('pictures/media_tempo')
    # plt.close()


def generate_awaretime_bins(dataset):
    df = dataset
    df['Percepcao_2'] = ''

    # df[(df.Percepcao > 7) & (df.SomCarro == 'Nao')].groupby(['DirecaoCarro', 'Consciente']).size().unstack().plot(kind='bar', stacked=True)
    # df[(df.Percepcao > 12.6) & (df.SomCarro == 'Sim')].groupby(['DirecaoCarro', 'Consciente']).size().unstack().plot(kind='bar', stacked=True)

    mean_aware = get_mean_aware_value_by_direction_soundtype(dataset)
    generate_line_plot(mean_aware)
    for car_dir, sound in mean_aware.items():
        for sound_type, mean_value in sound.items():
            mask = (df.DirecaoCarro == car_dir) & (df.SomCarro == sound_type) & (df.Percepcao <= mean_value)
            df.loc[mask, 'Percepcao_2'] = 'Presente'

            mask = (df.DirecaoCarro == car_dir) & (df.SomCarro == sound_type) & (df.Percepcao > mean_value)
            df.loc[mask, 'Percepcao_2'] = 'Tardia'

    mask = (df.Consciente == 'Inconsciente')
    df.loc[mask, 'Percepcao_2'] = 'Ausente'

    df.drop('Percepcao', axis=1, inplace=True)
    df.rename(columns={'Percepcao_2': 'Percepcao'}, inplace=True)

    count = df.groupby(['Consciente', 'Percepcao']).size().unstack().plot(kind='bar', stacked=True)
    # plt.title("Histograma Percepcao em 3 classes")
    # plt.xlabel("Quantidade")
    # plt.ylabel("Percepcao")

    # plt.savefig('pictures/percepcao')
    # plt.close()



dataset = dataset[dataset.Percepcao <= 26]
generate_awaretime_bins(dataset)
SPLIT = int(len(dataset) * 0.7)
TRAIN = dataset[:SPLIT].copy()
TEST = dataset[SPLIT:].copy()


def generate_histogram(dataset):
    # Histograma utilizado
    plt.title("Histograma Percepcao - 5 Intervalos")
    plt.xlabel("Tempo")
    plt.ylabel("Tamanho")

    plt.hist(dataset.Percepcao, bins=bins, edgecolor="k")
    plt.xticks(bins)
    plt.savefig('pictures/Figura Utilizada - sem retirar')
    plt.close()


def generate_awaretime_histograms(dataset):
    for ran in (4, 6, 8, 10, 12, 14, 16, 18, 20):
        for function in (pd.qcut, pd.cut):
            ser, bins = function(dataset.Percepcao, ran, labels=False, retbins=True)

            plt.title("Histograma Percepcao {} Intervalos".format(str(ran)))
            plt.xlabel("Tempo")
            plt.ylabel("Tamanho")

            plt.hist(dataset.Percepcao, bins=bins, edgecolor="k")
            plt.xticks(bins)

            if function == pd.cut:
                text = 'Intervalos iguais'
            else:
                text = 'Frequenciais iguais'

            plt.savefig('pictures/Figura {}-{}'.format(str(ran), text))
            plt.close()


def apply_k_means(dataset):
    dataset_2 = dataset.copy()
    dataset_2 = pd.get_dummies(dataset_2, columns=['Aware', 'AppDistraction', 'CarSound', 'CarDirection'])
    columns = [
        'Aware_Aware',
        'Aware_Unaware',
        'AppDistraction_Yes',
        'AppDistraction_No',
        'CarSound_Off',
        'CarSound_On',
        'CarDirection_Back',
        'CarDirection_Front',
        'CarDirection_Left',
        'CarDirection_Right',
        'AwareTime'
    ]

    dataset_2_std = stats.zscore(dataset_2[columns])

    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataset_2_std)
    labels = kmeans.labels_

    dataset_2['clusters'] = labels
    columns.extend(['clusters'])

    print(dataset_2[columns].groupby(['clusters']).mean())

    sns.lmplot('AwareTime', 'Aware_Aware',
           data=dataset_2,
           fit_reg=False,
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
    plt.title('Clusters AwareTime vs Aware_Aware')
    plt.xlabel('AwareTime')
    plt.ylabel('Aware_Aware')
    plt.show()



replacer = {
    'DistracaoApp': {
        'Nao': 0,
        'Sim': 1
    },
    'DirecaoCarro': {
        'Tras': 0,
        'Direita': 1,
        'Esquerda': 2,
        'Frente': 3
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



# count = pd.value_counts(df['Consciente'].values, sort=True)
# count.plot.bar()
# plt.title("Histograma Consciente")
# plt.xlabel("Consciente")
# plt.ylabel("Quantidade")
# 
# plt.xticks(rotation=0)
# plt.savefig('pictures/contagem_consciente')
# plt.close()

