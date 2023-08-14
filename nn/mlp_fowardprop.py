import numpy as np
from random import random
# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make some predictions

class MLP:
    def __init__(self, num_inputs=3,hidden_layers=[3,3],num_outputs=2):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        # initiate random weights
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1]) # to check
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i],layers[i+1]))
            derivatives.append(a)
        self.derivatives = derivatives
    
    def foward_propagate(self, inputs):

        # the input layer activation is just the input itself
        activations = inputs
        self.activations[0] = inputs
        
        # iterate throught the network layers
        for i, w in enumerate(self.weights):
            # calculate the net input
            net_inputs = np.dot(activations, w)
            # calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        return activations
    
    def back_propagate(self, error, verbose=False):
        # formulas
        # dE/dW_i = (y-a_[i+1]) s'(h_[i+1])a_i
        # s'[h_[i+1]] = s(h_[i+1])(1-s(h_[i+1]))
        # s(h_[i+1]) = a_[h+1]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            #print(delta_reshaped)
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0],-1)
            #print(current_activations_reshaped)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)#, delta_reshaped)
            error = np.dot(delta,self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
        
        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            #print("Original W{} {}".format(i, weights))

            derivatives = self.derivatives[i]

            weights += derivatives * learning_rate
            #print("Updated W{} {}".format(i, weights))
    
    def train(self,inputs, targets, epochs, learning_rate):
        for i in range(epochs):
            sum_errors = 0
            for j, input in enumerate(inputs):
                target = targets[j]
                output = self.foward_propagate(input)

                error = target - output
                self.back_propagate(error)

                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target,output)
            
            print("Error: {} at epoch {}".format(sum_errors/len(inputs),i+1))
        
        print("Training complete")
        print("=========")

            #for input, target in enumerate(zip(inputs, targets)):
                # foward propagation
            #    output = self.foward_propagate(input)
                # calculate error
            #    error = target - output
                # back propagation
            #    self.back_propagate(error)
                # apply gradient descent
            #    self.gradient_descent(learning_rate)

            #    sum_error += self._mse(target,output)
            
            # report error
            #print("Error: {} at epoch {}".format(sum_error/len(inputs),i))

    def _mse(self, target, output):
        return np.average((target-output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)                  
            
    
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    

if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in inputs])

    # create an MLP
    mlp = MLP(2,[5],1)

    # train our mlp
    mlp.train(inputs,targets, 100, 0.1)

    # create dummy data
    input = np.array([0.3,0.1])
    target = np.array([0.4])

    output = mlp.foward_propagate(input)
    print()
    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))

    # create some inputs
    #inputs = np.random.rand(mlp.num_inputs)
    #input = np.array([0.1,0.2])
    #target = np.array([0.3])
    # perform forward prop
    #output = mlp.foward_propagate(input)
    # calculate the target
    #error = target - output
    # backpropagation
    #mlp.back_propagate(error, verbose=True)
    # apply gradient descent 
    #mlp.gradient_descent(learning_rate=0.7)
    # print the results
    #print("The network input is: {}".format(inputs))
    #print("The network output is: {}".format(outputs))


    


            




