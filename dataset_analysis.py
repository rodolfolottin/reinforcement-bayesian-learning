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

