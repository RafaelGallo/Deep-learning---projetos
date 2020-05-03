#Função ativação para redes neurais A.i
#Função stepFunctio

import numpy as np

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

teste = stepFunction(30)
stepFunction(0)
stepFunction(-1)


#Função SigmoidFunctio
def sigmoidFunction(soma):
    return 1/ (1 + np.exp(-soma))

teste = sigmoidFunction(-1)
sigmoidFunction(-1)
teste = sigmoidFunction(0.358)
sigmoidFunction(0.358)

#Função Tahn-Function
def tahnFunction(soma):
    return (np.exp(soma)) / (np.exp(soma) + np.exp(-soma))

teste = tahnFunction(-0.358)
tahnFunction(-0.358)


#Função relu-Function
def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

teste = reluFunction(0.358)
reluFunction(0.358)


def linearFunction(soma):
    return soma

linearFunction(-0.358)
teste = linearFunction(-0.358)

def softmaxFunction(x):
    ex = np.exp(x)
    return ex/ex.sum()

valores = [5.0, 2.0, 1.3]
print(softmaxFunction(valores))




