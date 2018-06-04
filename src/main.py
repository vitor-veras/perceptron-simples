# coding: utf-8
# Author: Vitor Veras
import argparse
import perceptron
import dataManipulation

iris_path = './../samples/iris.data'
artificial1_path = './../samples/artificial1.data'

def main():

    # Parametros
    str = "Single layer Perceptron\n\t Author: Vitor Veras.\t Default arguments: -r=10 ; -d=0 ; -e=200 ; -p=0.8 ; -l=0.001"
    parser = argparse.ArgumentParser(description=str)
    parser.add_argument("-r", type=int, default=10, help="Number of executions")
    parser.add_argument("-d", type=int, default=0, help="Data set(0 = Iris - 1 = Artificial 1)", choices=range(0,2))
    parser.add_argument("-e", type=int, default=200, help="Number of epochs")
    parser.add_argument("-p", type=float, default=0.8, help="Trainning proportion")
    parser.add_argument("-l", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    if args.d==0:
        dm = dataManipulation.dataManipulation(iris_path, args.d)
    else:
        dm = dataManipulation.dataManipulation(iris_path, args.d)


    p = perceptron.Perceptron(dm.getData(), args.p, args.l, args.e)
    p.execution(args.r)


if __name__ == "__main__":
    main()
