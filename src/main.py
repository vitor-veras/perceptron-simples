# coding: utf-8
# Author: Vitor Veras
from src import perceptron
from src import dataManipulation

iris_path = './../samples/iris.data'
artificial1_path = './../samples/artificial1.data'

dm = dataManipulation.dataManipulation(artificial1_path, 1)
data = dm.getData()


def main():
    p = perceptron.Perceptron(data)
    p.execution(10)


if __name__ == "__main__":
    main()
