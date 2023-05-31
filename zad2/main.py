import numpy as np

def neuron(weight, goal, input, alpha, rep):
    for x in range(0,rep):
        pred=input*weight
        delta= 2 * (pred - goal) * input
        error=(pred-goal)**2
        weight=weight-(delta*alpha)
    return pred,round(error,10)

if __name__ == '__main__':
    output,error=neuron(0.5,0.8,2,0.1,5)
    print(output,error)
    output,error=neuron(0.5,0.8,2,0.1,20)
    print(output,error)


