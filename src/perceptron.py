# coding: utf-8
# Author: Vitor Veras
import numpy as np


class Perceptron():
    # w: weight list - lista de pesos
    # lr: learning rate - taxa de aprendizagem - default: 0.1
    # epochs: épocas de treinamento
    # proportion: proporção de treino/treino

    # CONSTRUTOR
    def __init__(self, data, proportion=0.8, learning_rate=0.001, epochs=200):
        self.data = data  # Base de dados
        self.x0 = [True, -1.0]  # Serve para indicar se teremos a inserção do X0 e se sim qual seu valor
        if self.x0[0]:
            d = []
            for i in range(len(self.data)):
                d.append(np.insert(self.data[i], 0, self.x0[1]))  # insere o valor de x0 para todos os padrões
            d = np.asarray(d)
            self.data = d
            self.w = self.resetWeight(self.data.shape[1])
        else:
            self.w = self.resetWeight(self.data.shape[1])

        self.proportion = proportion  # treina com 80% dos dados e testa com 20% dos dados
        self.lr = learning_rate  # Taxa de aprendizagem
        self.epochs = epochs  # Número máximo de épocas

    # FUNÇÃO DEGRAU
    # u = X . W(produto interno)
    # y=1 se u>0; y=0 se u<=0
    def step_fn(self, x):
        u = np.dot(self.w, x)
        if u < 0:
            y = 0
        else:
            y = 1
        return y

    # REGRA DE APRENDIZAGEM / AJUSTE DO VETOR W
    # w(t+1)=w(t) + (taxa_aprendizagem * erro_iteração)*x(t)
    def adjust(self, x, iteration_error):
        self.w = self.w + (self.lr) * (iteration_error) * x

    # ERRO NA ITERAÇÃO
    # d - y(esperado - saída da função degrau)
    def iterationError(self, x):
        y = self.step_fn(x)
        d = self.getLabel(x)
        if d != y:
            return d - y
        else:
            return 0

    # FUNÇÃO DE TREINO
    # globalError: erro global ao fim de uma época(soma dos erros das iterações)
    # self.epochs: numero maximo de épocas
    # data: base de dados de treino
    def training(self):
        data = self.data[0: int(len(self.data) * self.proportion)]  # utiliza apenas a proporção certa dos dados
        i = 1
        stop = False
        while not stop:
            globalError = 0.0
            np.random.shuffle(data)  # shuffle entre épocas
            for x in data:
                iteration_err = self.iterationError(x)  # Verifica o erro da iteração
                if iteration_err != 0:  # Se ocorrer erro
                    self.adjust(x, iteration_err)  # Ajuste de W
                    globalError += abs(iteration_err)
            i += 1
            if globalError == 0.0 or i >= self.epochs:  # Teste de parada
                stop = True
        return i  # Retorna o numero de epocas que parou

    # TESTES
    def test(self):
        data = self.data[int(len(self.data) * self.proportion):]  # utiliza apenas a proporção certa dos dados
        cn_mtx = [[0, 0], [0, 0]]  # Matriz de confusão
        cn_mtx = np.asarray(cn_mtx)
        for x in data:
            y = self.step_fn(x)
            d = self.getLabel(x)
            cn_mtx[y][d] += 1
        acc = np.trace(cn_mtx)  # acc recebe a soma da diagonal principal, ou seja, dos acertos
        print("Matriz de confusão: \n", cn_mtx)
        return acc / len(data)  # Retorna taxa de acerto([0,1])

    # REALIZAÇÃO
    # Faz uma realização completa, que consiste em:
    #   - Treino
    #   - Testes
    def execution(self, times):
        dp = 0  # desvio padrão
        acc_tx = []  # lista que salva a taxa de acerto de cada realização
        print("### PERCEPTRON SIMPLES ###")
        print("PARÂMETROS: ")
        print("\t W0: ", self.w)
        print("\t Taxa de aprendizagem: ", self.lr)
        print("\t Número máximo de épocas: ", self.epochs)
        print("\t Proporção treinamento/teste: ", (self.proportion) * 100, "/", (1 - self.proportion) * 100)
        print("\t Total de realizações: ", times, "\n")

        for i in range(1, times + 1):
            print("### REALIZAÇÃO ", i, "###")
            np.random.shuffle(self.data)  # shuffle entre realizações
            self.w = self.resetWeight(self.data.shape[1])  # reseta o vetor de pesos entre realizações
            print("### FASE DE TREINAMENTO ###")
            tr = self.training()
            print("Vetor W final: ", self.w)
            print("Número de épocas: ", tr)
            print("### FASE DE TESTES ###")
            tx = self.test()
            print("Taxa de acerto: ", tx, "\n")
            acc_tx.append(tx)

        accuracy = (sum(acc_tx) / times)  # acurácia entre [0,1]
        # Cálculo do desvio padrão
        for j in range(len(acc_tx)):
            dp += abs(accuracy - acc_tx[j])
        dp = dp / times
        accuracy *= 100  # acurácia em porcentagem
        print("DESVIO PADRÃO: ", dp)
        print("ACURÁCIA: ", accuracy)
        print("### FIM PERCEPTRON ###")

    # Retorna a ultima coluna de certo x
    def getLabel(self, x):
        return int(x[x.size - 1])

    # Serve para resetar o vetor de pesos com valores aleatórios entre [0,1]
    def resetWeight(self, n):
        return np.random.rand(1, n)
