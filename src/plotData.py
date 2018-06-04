# coding: utf-8
# Author: Vitor Veras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Plot
# arquivo de utilidade para plotar os padrões
def plot1(data):
    df = pd.DataFrame(data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    plt.title("Iris Data Set")
    g = sns.pairplot(data=df, hue="species", vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                     markers=["o", "^"])
    g.fig.subplots_adjust(top=0.1)
    g.fig.suptitle('Iris DataSet', fontsize=16)
    plt.show()


def plot2(data):
    df = pd.DataFrame(data, columns=['x', 'y', 'Label'])
    sns.lmplot(x='x', y='y', data=df, fit_reg=False, hue='Label', markers=["o", "*"])
    plt.title("Artificial I Data Set")
    plt.show()
