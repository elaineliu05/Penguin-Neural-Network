import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# activation function
def sigma(z):
    #sig_z = max(0, z)
    sig_z = 1/(1 + np.exp(-z))
    return sig_z
# derivative of activation function
def sigma_derivative(a):
    sig_deriv_a = a * (1 - a)
    return sig_deriv_a
# initialize matrix with random values
def initialize_random_weights(i,j):
    matrix = np.random.randn(i, j)
    return matrix
# cost function
def cost_function(y, y_hat):
    error = 1/(2*len(y))*np.mean((y - y_hat)**2) #mean squared error, they are vectors
    return error

# feed forward
def feed_forward(x): #how you compute y hat
  #Layer 1
  z1 = np.dot(W1, x) + b1
  a1 = sigma(z1)

  #Layer 2
  z2 = np.dot(W2.T, a1) + b2
  a2 = sigma(z2)

  return a1, a2

# gradient
def gradient(x, y, a1, a2):
    #calculates change in weight biases
    diff = a2 - y
    db2 = diff * sigma_derivative(a2)
    db1 = np.dot(W2, db2) *sigma_derivative(a1)

    #calculates change in weight matrices
    dW2 = np.dot(db2, a1.T)
    dW1 = np.dot(db1, x.T)

    return dW1, db1, dW2, db2

# training network
def train(inputs, outputs, learning_rate, epochs):
    global error_df
    error_df = pd.DataFrame(columns = ['epoch number', 'epoch calculated error'])
    for epoch in range(epochs):
        sum_error = 0
        for x,y in zip(inputs, outputs):
             reshape x to correct vector size
            x = np.reshape(x, (3,1))
            # edit: feed forward
            a1, a2 = feed_forward(x)
            # edit: calculate error using cost function
            error = cost_function(y, a2)
            sum_error = sum_error+error
            # edit: backpropogate using gradient function
            dW1, db1, dW2, db2 = gradient(x, y, a1, a2)
            # set to global variable to update values outside of function
            global W1
            global b1
            global W2
            global b2
            # edit: reassign matrices
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2.T
            b2 -= learning_rate * db2
        row = {'epoch number': epoch, 'epoch calculated error': sum_error/len(x)}
        error_df.loc[len(error_df)] = row
        if (epoch % 50 == 0):
          print(W1, W2)
    print(error_df)

#testing
def test(inputs, outputs):
    count_correct = 0
    for x,y in zip(inputs,outputs):
        x = np.reshape(x, (3,1))
        # edit: feed forward
        a1, a2 = feed_forward(x)
        # edit: determine if result is correct & update count_correct value
        if (a2 >= 0.5) & (y == 1):
          count_correct = count_correct + 1
        elif (a2 <= 0.5) & (y == 0):
          count_correct = count_correct + 1
    # edit: calculate percent_correct = 100*count_correct/total_count
    percent_correct = 100*count_correct/len(inputs)

    return percent_correct

#Running

# Loading data
X = pd.read_csv('penguins1.csv', usecols=range(1,4))
X = X.to_numpy()
Y = pd.read_csv('penguins2.csv', usecols=range(1,2))
Y = Y.to_numpy()

# normalize X dataset
X = normalize(X, axis=1, norm='l2')
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
#initializing weights
#Weights
W1 = initialize_random_weights(3,3)
W2 = initialize_random_weights(3,1)

#Biases
b1 = initialize_random_weights(3,1)
b2 = initialize_random_weights(1,1)

learning_rate = .1
epochs = 500
# train and test
train(x_train, y_train, learning_rate, epochs)
test(x_test, y_test)

#Visualizing accuracy
error_df.plot(x = "epoch number", y = "epoch calculated error")
