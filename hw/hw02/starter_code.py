import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test - test set, a 2D numpy array of size (num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    train_min = np.min(train, axis = 0)
    train_max = np.max(train, axis = 0)
    train_range = train_max - train_min
    train_normalized = (train - train_min)/train_range
    test_normalized = (test - train_min)/train_range
    return train_normalized,test_normalized


#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the average square loss, scalar
    """
    loss = 0
    loss = np.dot(X, theta) - y
    return 0.5 * np.sum(loss ** 2) / X.shape[0]

#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    #TODO
    y_predict = np.dot(X, theta)
    loss_gradient = np.dot(X.T, np.subtract(y_predict, y)) / X.shape[0]
    return loss_gradient
 

#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO
    true_gradient = compute_square_loss_gradient(X, y, theta)  # The true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features)  # Initialize the gradient we approximate
    # TODO
    basis_vectors = np.identity(X.shape[1])
    directional_change = epsilon * basis_vectors
    for i in range(len(basis_vectors)):
        approximation = (compute_square_loss(X, y, theta + directional_change[i]) - compute_square_loss(X, y, theta - directional_change[i])) / (2 * epsilon)
    if np.linalg.norm(approximation - true_gradient) >= tolerance:
        return False
    else:
        return True


#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000,grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step (num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array, (num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    # Initialize theta_hist
    theta_hist = np.zeros((num_step + 1, num_features))
    # Initialize loss_hist
    loss_hist = np.zeros(num_step + 1)
    # Initialize theta
    theta = np.zeros(num_features)
    for step_i in range(num_step + 1):
        theta_hist[step_i, ] = theta
        loss_hist[step_i] = compute_square_loss(X, y, theta)
        if grad_check:
            if not grad_checker(X, y, theta):
                print(f"Gradiant Fails! \n Alpha: {alpha}; Step: {step_i}")
                break
        theta -= compute_square_loss_gradient(X, y, theta) * alpha
    return theta_hist, loss_hist


#######################################
### Backtracking line search
#Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
#TODO
def backtracking_line_search(X, y, theta):
    sigma = 0.01
    beta = 0.5
    k = 0
    alpha = 1 #step_size
    while(True):
        loss_gradient = compute_square_loss_gradient(X,y,theta)
        theta_next = theta - alpha * loss_gradient
        if (compute_square_loss(X,y,theta) - compute_square_loss(X,y,theta_next) >= sigma * alpha * np.dot(loss_gradient.T,loss_gradient)):
            break
        else:
            alpha = beta * alpha
            theta = theta_next
            k = k+1
    print(alpha)
            
            
# plot - different alpha
def step_size_plot(X, y):
    alphas = [0.01, 0.05, 0.1, 0.5, 0.0625]
    color = ["red", "green","blue", "yellow", "pink"]
    for i,alpha in enumerate(alphas):
        start = time.time()
        theta_hist, loss_hist = batch_grad_descent(X, y, alpha, num_step=1000)
        end = time.time()
        print("the" + str(i) + "alpha test cost:" + str(end - start))
        plt.plot(loss_hist, label=f"Alpha={alpha}", color = color[i])
    plt.yscale('log') 
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg = 0.01, num_epoch=1000):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch (num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size (num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist
    #TODO
    step_size = 1
    for i in range(num_epoch):
       
        for j in range(num_instances):
            theta_hist[i,j] = theta
            regulariztion_loss = lambda_reg * np.dot(theta.T,theta)
            loss_hist[i,j] = compute_square_loss(X, y, theta) + regulariztion_loss
            gradient_part = 2*(np.dot(theta.T, X[j])-y[j])*X[j] + 2*lambda_reg*theta
            
            if alpha == "1/sqrt(t)":
                theta = theta - 0.1/step_size *gradient_part
            elif alpha == "1/t":
                theta = theta - 0.1/np.sqrt(step_size)*gradient_part
            else:
                theta = theta - alpha*gradient_part
            step_size += 1
    return theta_hist, loss_hist
            



def test_SGD(X, y):
    alphas = ["1/t","1/sqrt(t)",0.0005,0.001, 0.01]
    fig = plt.figure(figsize = (20,8))
    plt.subplot(222)
    for alpha in alphas:
        [theta_hist, loss_hist] = stochastic_grad_descent(X, y, alpha, num_epoch = 5)
        plt.plot(np.log(loss_hist.ravel()), label = 'alpha:'+str(alpha))
    plt.legend()
    

def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(x_train, x_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # Add bias term
    # TODO
    theta = np.ones(X_train.shape[1])
    loss = compute_square_loss(X_train, y_train, theta)
    loss_gradient = compute_square_loss_gradient(X_train, y_train, theta)
    print(loss)
    print(loss_gradient)
    
    #small dataset to check the function
    X_small_data = np.random.rand(2,2)
    y_small_data = np.random.rand(2,1)
    theta_small_data = np.random.rand(2,1)
    print(X_small_data)
    print(y_small_data)
    print(theta_small_data)
    print(compute_square_loss(X_small_data, y_small_data, theta_small_data))
    print(compute_square_loss_gradient(X_small_data, y_small_data, theta_small_data))
    
    
    #chech batch GD
    loss_return, theta_return = batch_grad_descent(X_train, y_train)
    print("loss return:", str(loss_return))
    print("theta return:", str(theta_return))

    step_size_plot(X_train, y_train)
    backtracking_line_search(X_train,y_train,np.ones(X_train.shape[1]))
    
    theta_hist, loss_hist = stochastic_grad_descent(X_train, y_train)
    print(theta_hist)
    print(loss_hist)
    test_SGD(X_train, y_train)

  
    
    
if __name__ == "__main__":
    main()
