import numpy as np
import pickle

''' An implementation of an MLP with a single layer of hidden units. '''
class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.
        
        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.
        
        Note: a1 and z1 can be used for caching during backprop/evaluation.
        
        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units

        self.b1 = np.zeros((self.hidden_units, ))
        self.b2 = np.zeros((self.dout, ))
        self.W1 = 2*(np.random.random((self.hidden_units, self.din)) - 0.5)
        self.W2 = 2*(np.random.random((self.dout, self.hidden_units)) - 0.5)


    def save(self,filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    def softmax(self, y):
        y_ = []
        y_denominator = 0
        if len(y.shape) == 2:
            y_denominator = np.zeros((np.shape(y)[1],))
        for i in range(self.dout):
            y_denominator+=np.exp(y[i])
        for i in range(self.dout):
            y_col = np.exp(y[i])/y_denominator
            y_.append(y_col)
        return np.array(y_)

    def eval(self, xdata):
        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.
        '''
        b1N = np.array([self.b1 for i in range(np.shape(xdata)[1])]).transpose()
        b2N = np.array([self.b2 for i in range(np.shape(xdata)[1])]).transpose()
        a1 = np.matmul(self.W1, xdata)+b1N
        z1 = np.tanh(a1)
        ydata = np.matmul(self.W2, z1)+b2N
        ydata = self.softmax(ydata)
        return np.array(ydata)

    def forward_prop(self, x):
        self.a1 = np.matmul(self.W1, x)+self.b1
        self.z1 = np.tanh(self.a1)
        y = np.matmul(self.W2, self.z1)+self.b2
        y = self.softmax(y)
        '''print(x, "W1", self.W1, "matmul",np.matmul(self.W1, x), self.a1, self.z1, y)
        print("++++++++++")'''
        return np.array(y).transpose()

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. ''' 
        for x, y in zip(xdata.transpose(), ydata.transpose()):
            grads = self.grad(x, y)
            self.W1 += learn_rate*grads[0]
            self.b1 += learn_rate*grads[1]
            self.W2 += learn_rate*grads[2]
            self.b2 += learn_rate*grads[3]
        
    def grad(self, x, y):
        ''' Return a tuple of the gradients of error wrt each parameter. 

        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        t = self.forward_prop(x)
        # delta_k has dimension (dout x 1)
        delta_k = y - t
        # delta_j has dimension (hidden_units x 1)
        delta_j = (1-self.z1**2)*np.matmul(delta_k.transpose(), self.W2).transpose()
        # dEdW1 has dimension (hidden_units x din)
        dEdW1 = np.matmul(delta_j.reshape((self.hidden_units, 1)), x.reshape((self.din, 1)).transpose())
        # dEdb1 has dimension (hidden_units x 1)
        dEdb1 = delta_j
        # dEdW2 has dimension (dout x hidden_units)
        dEdW2 = np.matmul(delta_k.reshape((self.dout, 1)), self.z1.reshape((self.hidden_units, 1)).transpose())
        # dEdb1 has dimension (dout x 1)
        dEdb2 = delta_k
        #printing statements for debugging purposes:
        #if np.all((self.z1**2 == 1)):
        '''if np.argmax(t) != np.argmax(y):
            print(x)
            print(t, y)
            print(self.a1,"\n" ,self.z1,"\n" ,self.W1,"\n" ,self.W2,"\n" ,self.b1,"\n" ,self.b2)
            print(delta_k,"\n" , delta_j,"\n" , dEdW1,"\n" , dEdb1,"\n" , dEdW2,"\n" , dEdb2)
            print("==========")
            print()'''
        return (dEdW1, dEdb1, dEdW2, dEdb2)


