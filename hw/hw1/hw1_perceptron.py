# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:58:35 2019

@author: ruby_
"""

#lecture
"""
formulate a proble,
gather dara
explore data
determine a model
evaluate findings

training set
validation set
testing set


"""

import matplotlib.pyplot as plt
import math
from collections import Counter
import numpy as np 
import matplotlib as mpl


"""
question one
split the training data (i.e. spam train.txt) into a training and validate set, 
putting the first 4000 emails into the training set
putting the last 1000 emails into the validation set. 
when putting each email into the training set and validation set, split each letter
then seprate the first letter which is 0 or 1, the classification of them
"""
def get_ads():
    training_set = []
    validation_set = []
    training_data_classifications = []
    validation_data_classifications = []
    with open("spam_train.txt") as training_data_file:
        for i,line in enumerate(training_data_file):
            if i < 4000:
                training_set.append(line.split())
                training_data_classifications.append(training_set[i].pop(0))
            if i >= 4000:
                validation_set.append(line.split())
                validation_data_classifications.append(validation_set[i-4000].pop(0))
            
    return training_set,training_data_classifications, validation_set, validation_data_classifications

training_set,training_data_classifications, validation_set, validation_data_classifications = get_ads()
            
"""
question two
Transform all of the data into feature vectors. 
Build a vocabulary list using only the 4000 e-mail training set by ﬁnding all words that occur across the training set. 
Ignore all words that appear in fewer than X = 30 emails, so we need to use dict in python, 
to trans all lines in training_set in to dict, to know the number of the words appear
e-mails of the 4000 e-mail training set – this is both a means of preventing overﬁtting and of improving scalability. 

"""


    
def get_vocabulary_list(X):
    vocabulary_list = []
    """
    using dict.fromkeys() to remove the words appear many times in one line
    example:
        seq = ('Google', 'Runoob', 'Taobao','Google', 'Runoob', 'Taobao')
        >>> dict = dict.fromkeys(seq)
        >>> dict
        {'Google': None, 'Runoob': None, 'Taobao': None}
    """
    
    for line in training_set:
        vocabulary_list += (list(dict.fromkeys(line))) 
    
    # using dict to compute the number that word appear in different emails 
    # if numbers bigger than 30 then store in final_vocabulary_list
        
    counts = {}
    for word in vocabulary_list:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    
    final_vocabulary_list = []
    for word in counts:
        if counts[word] >= X:
            final_vocabulary_list.append(word)
    return final_vocabulary_list



final_vocabulary_list = get_vocabulary_list(30)
        
"""
For each email, transform it into a feature vector 
x where the ith entry, xi, is 1 if the ith word in the vocabulary occurs in the email, and 0 otherwise.
"""
def get_feature_vectors(training_set):
    feature_vectors = []
    feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]
    # adding bach the classification in the first space
    feature_vectors.insert(0,training_data_classifications)
    return feature_vectors

feature_vectors = get_feature_vectors(training_set)

"""
question three
Implement the functions perceptron train(data) and perceptron test(w, data). 
The function perceptron train(data) trains a perceptron classiﬁer using the examples provided to the function, 

For the corner case of w·x = 0, predict the +1 (spam) class. 

The function perceptron test(w, data) should take as input the weight vector w
 (the classiﬁcation vector to be used) and a set of examples. 

return :
    w: the ﬁnal classiﬁcation vector, theta
    k: the number of updates (mistakes) performed
    iter: the number of passes through the data, respectively
"""


def perceptron_train(data,data_classification):
    
    # seprate the classification from each data for further use
    # and the vector is already delete the first space which is label
    classifications = data_classification
    # change the label from 0 to -1, according to the instructor
    classifications = ['-1' if x=='0' else x for x in classifications]
    #print(classifications)
    # return items
    w = [0]*len(data[0])  #weight
    k = 0                 #number of update
    iter = 0              # mistakes
    finish = False # need a flag for the algorithm to stop

    while finish is False:
        finish = True
        # data = [[],[],...,[],[]]
        for t,vector in enumerate(data):
            activation = 0
            # vector = [,...,]
                #activation function
                #if activation >= 0 then return +1, else return -1
            activation = np.dot(w,vector)
            
            if activation * int(classifications[t]) <= 0 and np.sum(vector) > 0 or (activation == 0 and classifications[t] == '-1'):
                for i in range(0,len(vector)):
                    # update the weight
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                k = k + 1 # mistake count +1
                finish = False # till done equal to true, stop 
        iter = iter + 1
    print(iter)
    return w,k,iter   
    
    
def perceptron_test(w, data,data_classification):
    
    classifications = data_classification
    prediction_label = []
    count = 0
 
    for vector in data:
        activation = 0
        for i in range(0,len(vector)):
            activation += w[i]*vector[i]
        if activation >= 0:
            prediction_label.append('1')
        else:
            prediction_label.append('0')
    num = len(classifications)
    

    combine_label_classifications = zip(prediction_label,classifications)
    for i,j in combine_label_classifications:
        if i == j:
            count += 1
    # count is the number which is classified right
    return (num - count)/num


"""
question four
Train the linear classiﬁer using your training set. 
Test your implementation of perceptron test by running it with the learned parameters and the training data, 
making sure that the training error is zero. 
Next, classify the emails in your validation set.
"""

# adding bach the classification in the first space
def train_perceptron(feature_vectors):
    w,k,iter = perceptron_train(feature_vectors,training_data_classifications)
    error = perceptron_test(w,feature_vectors,training_data_classifications)
    print("Mistakes made while training the training data: ",k)
    print("Training error when testing the w and training data: ",error)
    return w

def validation_percetron(w, feature_vectors):
    # manage validation data same with question two
    # using the same w for the validation data
    error = perceptron_test(w,feature_vector_validation,validation_data_classifications)
    print("Validation error with the former w and validation_data_classification: ",error)

feature_vectors.pop(0)
w = train_perceptron(feature_vectors)
feature_vector_validation = []
feature_vector_validation = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in validation_set]
validation_percetron(w, feature_vector_validation)


"""
question five:
output the 15 words with the most positive weights.
each time get the maximum weights and pop that out the w list
"""
def get_highest_15(w,final_vocabulary_list):
    positive_weights = []
    w_array = np.array(w)
    argsort_w =np.argsort(w_array)
    index= argsort_w[::-1]
    for item in index[:15]:
        positive_weights.append(final_vocabulary_list[item])
    
    print(positive_weights)
   
get_highest_15(w,final_vocabulary_list)


"""
question six:
Implement the averaged perceptron algorithm,returns the average of all weight vectors considered during the algorithm. 
Averaging reduces the variance between the diﬀerent vectors
"""
def average_perceptron_train(data,data_calssification):
    classifications = data_calssification
    classifications = ['-1' if x=='0' else x for x in classifications]

    w = [0]*len(data[0])
    #average_w = []
    k = 0
    iter = 0
    done = False
    cache_w = [0]*len(data[0])
    count = 1
    while not done:
        done = True
        for t,vector in enumerate(data):
            activation = 0
            activation = np.dot(w,vector)
            
            if activation * int(classifications[t]) <= 0 and np.sum(vector) > 0 or (activation == 0 and classifications[t] == '-1'):
                for i in range(0,len(vector)):
                    # update the weight
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                    cache_w[i] = cache_w[i] + count*(vector[i]*int(classifications[t]))
                k = k + 1
                done = False
            count += 1
        #average_w.append(w)
        iter = iter + 1
        cache_w = np.array(cache_w)
        average_change = np.array(w) - (1/count)*cache_w
    return list(average_change),k,iter

def average_perceptron_train_try(data,data_calssification):
    classifications = data_calssification
    classifications = ['-1' if x=='0' else x for x in classifications]

    w = [0]*len(data[0])
    #average_w = []
    k = 0
    iter = 0
    done = False
    cache_w = [0]*len(data[0])
    count = 1
    while not done:
        done = True
        for t,vector in enumerate(data):
            activation = 0
            activation = np.dot(w,vector)
            
            if activation * int(classifications[t]) <= 0 and np.sum(vector) > 0 or (activation == 0 and classifications[t] == '-1'):
                for i in range(0,len(vector)):
                    # update the weight
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                k = k + 1
                done = False
            count += 1
            cache_w = np.array(cache_w)
        #average_w.append(w)
        iter = iter + 1
        
        cache_w += np.array(w)
        average_change = cache_w/(iter*4000)
    return list(average_change),k,iter

def train_average_perceptron(feature_vectors,training_data_classifications,feature_vector_validation,validation_data_classifications):
    w,k,iter = average_perceptron_train_try(feature_vectors,training_data_classifications)
    error_average_train = perceptron_test(w,feature_vectors,training_data_classifications)
    print("Mistakes made while training the training data with the average perceptron algoright:", k)
    print("Training error when teating the w and training data:", error_average_train)
    print("the number passes throught:", iter)
    error_average_validate = perceptron_test(w,feature_vector_validation,validation_data_classifications)
    print("Validation error with the former w and validation_data_classification ( average percetron): ",error_average_validate)

train_average_perceptron(feature_vectors,training_data_classifications,feature_vector_validation,validation_data_classifications)


"""
question severn
Add an argument to both the perceptron and the averaged perceptron 
that controls the maximum number of passes over the data. 
This is an important hyperparameter because for large training sets, 
the perceptron algorithm can take many iterations just changing a small subset of the point --
leading to overfitting.
"""
def perceptron_train_with_argument(data,data_classification,max_iterations):
    # seprate the classification from each data for further use
    # and the vector is already delete the first space which is label
    classifications = data_classification
    # change the label from 0 to -1, according to the instructor
    classifications = ['-1' if x=='0' else x for x in classifications]
    #print(classifications)
    # return items
    w = [0]*len(data[0])  #weight
    k = 0                 #number of mistakes
    iter = 0              #update
    

    # run 10 rounds and whole 40000 passes
    while iter < max_iterations:
    # data = [[],[],...,[],[]]
        for t,vector in enumerate(data):
            activation = 0
            activation = np.dot(w,vector)
            if activation * int(classifications[t]) <= 0 and np.sum(vector) > 0 or (activation == 0 and classifications[t] == '-1'):
                for i in range(0,len(vector)):
                    # update the weight
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                    
                k = k + 1 # mistake count +1
        iter = iter + 1
        
    return w,k,iter


    
def perceptron_train_averaged_with_argument(data,data_classification,max_iterations):
    classifications = data_classification
    classifications = ['-1' if x=='0' else x for x in classifications]

    w = [0]*len(data[0])
    k = 0
    iter = 0
    cache_w = [0]*len(data[0])
    count = 1
    while iter < max_iterations:
        for t,vector in enumerate(data):
            activation = 0
            activation = np.dot(w,vector)
            if activation * int(classifications[t]) <= 0 and np.sum(vector) > 0 or (activation == 0 and classifications[t] == '-1'):
                for i in range(0,len(vector)):
                    # update the weight
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                    cache_w[i] = cache_w[i] + count*(vector[i]*int(classifications[t]))
                k = k + 1
            count += 1
        #average_w.append(w)
        iter = iter + 1
        cache_w = np.array(cache_w)
        average_change = np.array(w) - (1/count)*cache_w
    return list(average_change),k,iter
def perceptron_train_averaged_with_argument_try(data,data_classification,max_iterations):
    classifications = data_classification
    classifications = ['-1' if x=='0' else x for x in classifications]

    w = [0]*len(data[0])
    k = 0
    iter = 0
    cache_w = [0]*len(data[0])
    count = 1
    while iter < max_iterations:
        for t,vector in enumerate(data):
            activation = 0
            activation = np.dot(w,vector)
            if activation * int(classifications[t]) <= 0 and np.sum(vector) > 0 or (activation == 0 and classifications[t] == '-1'):
                for i in range(0,len(vector)):
                    # update the weight
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                k = k + 1
                done = False
            count += 1
            cache_w = np.array(cache_w)
            cache_w += np.array(w)
        #average_w.append(w)
        iter = iter + 1
        
        
    average_change = cache_w/(iter*len(data))
    return list(average_change),k,iter
"""
question eight:
Experiment with various maximum iterations on the two algorithms checking performance on the validation set. 
Optionally you can try to change X from question 2. Report the best validation error for the two algorithms
     
"""
def train_with_argument(feature_vectors,training_data_classifications,feature_vector_validation,validation_data_classifications):
    for i in range(1,12):
        # adding bach the classification in the first space
        w,k,iter = perceptron_train_with_argument(feature_vectors,training_data_classifications,i)
        error_train = perceptron_test(w,feature_vectors,training_data_classifications)
        print("number passes iteration",iter)
        print("error from training set:", error_train)
        # using the same w for the validation data
        error_validation = perceptron_test(w,feature_vector_validation,validation_data_classifications)
        print("Mistakes made while training the training data with the perceptron algorighm:", k)
        print("Validation error with the former w and validation_data_classification: ",error_validation)
        print("---")
        w,k,iter = perceptron_train_averaged_with_argument_try(feature_vectors,training_data_classifications,i)
        error_train_average = perceptron_test(w,feature_vectors, training_data_classifications)
        print("[average] Validation error with the former w and validation_data_classification: ",error_train_average)
        # using the same w for the validation data
        error_validation_average = perceptron_test(w,feature_vector_validation,validation_data_classifications)
        print("[average] Mistakes made while training the training data with the perceptron algorighm:", k)
        print("[average] Validation error with the former w and validation_data_classification: ",error_validation_average)
        print("------")
        print([error_train,error_validation,error_train_average,error_validation_average])
        
print("------")
print(math.ceil(len(feature_vectors)/500) - 1)
#train_with_argument(feature_vectors,training_data_classifications,feature_vector_validation,validation_data_classifications)


error_count_from_one = [[0.0225, 0.03, 0.01325, 0.023],[0.01075, 0.023, 0.00675, 0.02],[0.00525, 0.022, 0.0045, 0.016],
                        [0.00225, 0.019, 0.0025, 0.016],[0.005, 0.023, 0.002, 0.016],[0.001, 0.017, 0.0015, 0.015],
                        [0.00025, 0.02, 0.00125, 0.013],[0.00025, 0.02, 0.00125, 0.014],[0.00025, 0.014, 0.00125, 0.016],
                        [0.0, 0.013, 0.001, 0.017],[0.0, 0.013, 0.00075, 0.016]]


error_train = [0]*11
error_validation = [0]*11
error_train_average = [0]*11
error_validation_average = [0]*11
average_error_perceptron = [0] * 11
average_error_averaged_perceptron = [0] * 11

for i,item in enumerate(error_count_from_one):
    error_train[i] = item[0]
    error_validation[i] = item[1]
    error_train_average[i] = item[2]
    error_validation_average[i] = item[3]
    average_error_perceptron[i] = [(item[0]+item[1])/2]
    average_error_averaged_perceptron[i] = (item[2]+item[3])/2
x = np.arange(1,12,1)
y1 = error_train
y2 = error_validation
y3 = error_train_average
y4 = error_validation_average
y5 = average_error_perceptron
y6 = average_error_averaged_perceptron

plt.plot(x, y1, color = "blue",linestyle="-",  marker ="^", label = "train error")
plt.plot(x, y2, color = "orange",linestyle="-",  marker = "s", label = "validation error")
plt.plot(x, y3, color = "green",linestyle="-",  marker ="^",label = "average train error")
plt.plot(x, y4, color = "red",linestyle="-",  marker = "s", label = "average validation error")
plt.plot(x, y5, color = "black",linestyle="-",  marker = "*")
plt.plot(x, y6, color = "black",linestyle="-",  marker = "+")

plt.legend(loc='upper right')
plt.xlabel("x")
plt.ylabel("error")
plt.show()



"""
question nine:
Combine the training set and the validation set and learn using the best of the configurations previously found. 
What is the error on the test set (i.e. spam$_$test.txt).
"""
testing_set_train_feature_vectors = feature_vectors + feature_vector_validation
testing_set_train_validation = training_data_classifications + validation_data_classifications     

def train_spam_test():
    test_set = []
    test_set_classification = []
    with open("spam_test.txt") as testing_data_file:
        for i,line in enumerate(testing_data_file):
            test_set.append(line.split())
            test_set_classification.append(test_set[i].pop(0))
    return test_set, test_set_classification

test_set, test_set_classification = train_spam_test()
test_feature_vectors = get_feature_vectors(test_set)
test_feature_vectors.pop(0)
w,k,iter = perceptron_train_with_argument(testing_set_train_feature_vectors,testing_set_train_validation,10)
error_test = perceptron_test(w,test_feature_vectors,test_set_classification)
w,k,iter = perceptron_train_averaged_with_argument(testing_set_train_feature_vectors,testing_set_train_validation,7)
error_test_average = perceptron_test(w,test_feature_vectors,test_set_classification)
print("error from test data:",error_test)
print("error with average perceptron:",error_test_average)








"""extra"""

def train_with_different_X(X):
    # create a new vocabulary list with different X
    final_vocabulary_list = get_vocabulary_list(X)
    # create new feature_vector with vocabulary_list
    for i in range(1,12):
        feature_vectors = []
        feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]
        w,k,iter = perceptron_train_with_argument(feature_vectors, training_data_classifications,i)
        error_train_validation = perceptron_test(w, feature_vector_validation, validation_data_classifications)
        print("with",iter)
        print("the error for validation data:", error_train_validation)
        print("---")

#train_with_different_X(50)