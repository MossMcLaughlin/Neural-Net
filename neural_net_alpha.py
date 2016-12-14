import numpy as np
import random
import data_loader
 
training_data, test_data = data_loader.get_data() 
 


def sigmoid(z):
    return 1/(1+np.exp(-z)) 

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
 


class Network(object):

    def __init__(self,layers):
        # layers is a list of the number of nodes for each layer of the net.
        self.num_layers = len(layers)
        self.layers = layers 
        '''For each layer, bias is a nx1 dim array (vector), 
        weight is nxm dim array (matrix)
        note no weight matrix for final layer (output)
        no bias vector for first layer(input). '''
        self.bias = [np.random.randn(i,1) for i in layers[1:]]
        self.weight = [np.random.randn(j,k) for j,k in zip(layers[1:],layers[:-1])]

    def feedforward(self,activation):
    # takes input (activation) and returns output.
        for b,w in zip(self.bias, self.weight):
            activation = sigmoid(np.dot(w,activation) + b) 
        return activation


    def gradient_descent(self,training_data,epochs,batch_size,learning_rate,track_results=False):
        data_length = len(training_data)
        num = len(test_data)
        for k in xrange(epochs):
            random.shuffle(training_data)

            data_batches = [training_data[l:l+batch_size] 
                for l in xrange(0, data_length, batch_size)]             
            for batch in data_batches: self.update(batch,learning_rate) 
	    if track_results == True : print("Epoch {}: {} / {} correct.\n".format(k,self.evaluate(test_data),num))
            else: print("Epoch {} complete".format(k))


	
    def update(self,batch,learning_rate):
        #Data is a list of tuples containing input data and desired output.
        partial_b = [np.zeros(i.shape) for i in self.bias] 
        partial_w = [np.zeros(j.shape) for j in self.weight]
        for data_in,desired_out in batch:
            delta_partial_b, delta_partial_w = self.backprop(data_in, desired_out)
            partial_b = [pb+dpb for pb, dpb in zip(partial_b, delta_partial_b)]
            partial_w = [pw+dpw for pw, dpw in zip(partial_w, delta_partial_w)]
#        print("partials WRT b, w",partial_b[0][0:5],partial_w[0][0][0:5])
#        print("example weight,bais: ",self.weight[0][0][0:5],self.bias[0][0:5]) 
        self.weight = [w-(learning_rate/len(batch)*pw) for w, pw in zip(self.weight, partial_w)]
        self.bias = [b-(learning_rate/len(batch)*pb) for b, pb in zip(self.bias, partial_b)]
#        print("updated w,b: ",self.weight[0][0][0:5],self.bias[0][0:5]) 

       
    def backprop(self,data_in,desired_out):
        partial_b = [np.zeros(i.shape) for i in self.bias] 
        partial_w = [np.zeros(j.shape) for j in self.weight]
        a = data_in
        # Create a list to store all activations and z vectors by layer
        # recall a = sigmoid(z) 
        a_list = [data_in]
        z_list = []
        for b, w in zip(self.bias,self.weight):
            z = np.dot(w,a)+b
            z_list.append(z)
            a = sigmoid(z)
            a_list.append(a)
        # Calculate error in last layer 
        output_error = self.partial_a(a_list[-1],desired_out) * sigmoid_prime(z_list[-1])
        partial_b[-1] = output_error
        partial_w[-1] = np.dot(output_error,a_list[-2].transpose())
        # Backprogigate that error
        for m in xrange(2,self.num_layers):
            output_error = np.dot(self.weight[-m+1].transpose(), \
            output_error) * sigmoid_prime(z_list[-m])        
            partial_b[-m] = output_error
            partial_w[-m] = np.dot(output_error,a_list[-m-1].transpose())
#        partial_w = np.array(partial_w)
#        print("partial_w",partial_w.shape,partial_w[-2])
        return (partial_b,partial_w)



    def partial_a(self,activation,desired_out):
        """    Returns the derivative of the quadratic cost function 
               C = 1/2 ((a-y)^2), dC/da = (a-y)     """
        return (activation-desired_out)
   

    def evaluate(self,data):
        results =[(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)



Net = Network([784,100,10,25,10])

Net.gradient_descent(training_data,25,30,5.0,track_results=True)







