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

import numpy as np
import matplotlib.pyplot as plt
import math
"""
question one
split the training data (i.e. spam train.txt) into a training and validate set, 
putting the first 4000 emails into the training set
putting the last 1000 emails into the validation set. 
when putting each email into the training set and validation set, split each letter
then seprate the first letter which is 0 or 1, the classification of them
"""

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
            
            
            
"""
question two
Transform all of the data into feature vectors. 
Build a vocabulary list using only the 4000 e-mail training set by ﬁnding all words that occur across the training set. 
Ignore all words that appear in fewer than X = 30 emails, so we need to use dict in python, 
to trans all lines in training_set in to dict, to know the number of the words appear
e-mails of the 4000 e-mail training set – this is both a means of preventing overﬁtting and of improving scalability. 

"""
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
    if counts[word] >= 30:
        final_vocabulary_list.append(word)
        
"""
For each email, transform it into a feature vector 
x where the ith entry, xi, is 1 if the ith word in the vocabulary occurs in the email, and 0 otherwise.
"""
feature_vectors = []
feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]
# adding bach the classification in the first space
feature_vectors.insert(0,training_data_classifications)

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
    
    # need a flag for the algorithm to stop
    finish = False

    while finish is False:
        finish = True
        # data = [[],[],...,[],[]]
        for t,vector in enumerate(data):
            activation = 0
            # vector = [,...,]
            for i in range(0,len(vector)):
                #activation function
                #if activation >= 0 then return +1, else return -1
                activation += w[i]*vector[i]
            if activation >= 0:
                predict = '1'
            else:
                predict = '-1'
            if predict != classifications[t]:
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
"""
# adding bach the classification in the first space
feature_vectors = []
feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]

w,k,iter = perceptron_train(feature_vectors,training_data_classifications)
error = perceptron_test(w,feature_vectors,training_data_classifications)
print("Mistakes made while training the training data: ",k)
print("Training error when testing the w and training data: ",error)

# manage validation data same with question two
feature_vector_validation = []
feature_vector_validation = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in validation_set]
# using the same w for the validation data
error = perceptron_test(w,feature_vector_validation,validation_data_classifications)
print("Validation error with the former w and validation_data_classification: ",error)

"""

"""
question five:
output the 15 words with the most positive weights.
each time get the maximum weights and pop that out the w list
"""
"""
positive_weights = []
for i in range(0,15):
    # after training with 4000 dataframe we got the final w, which training error is 0.0
    # pop each time the max weights, get the index of each time 
    w_max_index = w.index(max(w)) 
    positive_weights.append(final_vocabulary_list.pop(w.pop(w_max_index))) 
print("the most 15 positive weights words is", positive_weights)
"""

"""
question six:
Implement the averaged perceptron algorithm,returns the average of all weight vectors considered during the algorithm. 
Averaging reduces the variance between the diﬀerent vectors
"""
def average_perceptron_train(data,data_calssification):
    classifications = data_calssification
    classifications = ['-1' if x=='0' else x for x in classifications]

    w = [0]*len(data[0])
    average_w = []
    k = 0
    iter = 0
    done = False

    while not done:
        done = True
        for t,vector in enumerate(data):
            activation = 0
            for i in range(0,len(vector)):
                activation += w[i]*vector[i]
            if activation >= 0:
                predict = '1'
            else:
                predict = '-1'
            if predict != classifications[t]:
                for i in range(0,len(vector)):
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                k = k + 1
                done = False
        average_w.append(w)
        iter = iter + 1

    for each_w in average_w:
        for i in range(0,len(each_w)):
            w[i] += each_w[i]
            w[i] = w[i]/len(average_w)
     
    return w,k,iter
feature_vectors = []
feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]

w,k,iter = average_perceptron_train(feature_vectors,training_data_classifications)
error_average_train = perceptron_test(w,feature_vectors,training_data_classifications)
print("Mistakes made while training the training data with the average perceptron algoright:", k)
print("Training error when teating the w and training data:", error_average_train)
print("the number passes throught:", iter)

feature_vector_validation = []
feature_vector_validation = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in validation_set]
# using the same w for the validation data
error = perceptron_test(w,feature_vector_validation,validation_data_classifications)
print("Validation error with the former w and validation_data_classification ( average percetron): ",error)

"""
feature_vectors_valudation = []
feature_vectors_validation = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in validation_set]
w,k,iter = average_perceptron_train(feature_vectors_validation,validation_data_classifications)
error_average_validate = perceptron_test(w,feature_vectors_validation,validation_data_classifications)
print("Mistakes made while training the training data with the average perceptron algoright:", k)
print("Validation error with the former w and validation_data_classification: ", error_average_validate)
print("the number passes throught:", iter)
"""

"""
question severn
Add an argument to both the perceptron and the averaged perceptron 
that controls the maximum number of passes over the data. 
This is an important hyperparameter because for large training sets, 
the perceptron algorithm can take many iterations just changing a small subset of the point --
leading to overfitting.
"""
#横坐标是个数，纵坐标是error， 越多的训练，erro越小，然后画图，然后去看几次方的函数最合适
#第七问只是针对training set，然后画图，然后来看多少的data 是最低的error
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
            # vector = [,...,]
            for i in range(0,len(vector)):
                #activation function
                #if activation >= 0 then return +1, else return -1
                activation += w[i]*vector[i]
            if activation >= 0:
                predict = '1'
            else:
                predict = '-1'
            if predict != classifications[t]:
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
    average_w = []
    k = 0
    iter = 0

    while iter < max_iterations:
        for t,vector in enumerate(data):
            activation = 0
            for i in range(0,len(vector)):
                activation += w[i]*vector[i]
            if activation >= 0:
                predict = '1'
            else:
                predict = '-1'
            if predict != classifications[t]:
                for i in range(0,len(vector)):
                    w[i] = w[i] + (vector[i]*int(classifications[t]))
                k = k + 1
        average_w.append(w)
        iter = iter + 1

    for each_w in average_w:
        for i in range(0,len(each_w)):
            w[i] += each_w[i]
            w[i] = w[i]/len(average_w)

    return w,k,iter

def train_with_argument():
    error = [[0,0,0,0]]*11
    for i in range(1,12):
        feature_vectors = []
        feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]
        # adding bach the classification in the first space
        w,k,iter = perceptron_train_with_argument(feature_vectors,training_data_classifications,i)
        error_train = perceptron_test(w,feature_vectors,training_data_classifications)
        print("number passes iteration",iter)
        print("error from training set:", error_train)
        print("---")
        feature_vector_validation = []
        feature_vector_validation = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in validation_set]
        # using the same w for the validation data
        error_validation = perceptron_test(w,feature_vector_validation,validation_data_classifications)
        print("Mistakes made while training the training data with the perceptron algorighm:", k)
        print("Validation error with the former w and validation_data_classification: ",error_validation)
        print("---")
        feature_vectors = []
        feature_vectors = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in training_set]
        w,k,iter = perceptron_train_averaged_with_argument(feature_vectors,training_data_classifications,i)
        error_train_average = perceptron_test(w,feature_vectors, training_data_classifications)
        print("the number passes throught:", iter)
        print("Validation error with the former w and validation_data_classification: ",error_train_average)
        print("---")
        feature_vector_validation = []
        feature_vector_validation = [[1 if word in vector else 0 for word in final_vocabulary_list] for vector in validation_set]
        # using the same w for the validation data
        error_validation_average = perceptron_test(w,feature_vector_validation,validation_data_classifications)
        print("Mistakes made while training the training data with the perceptron algorighm:", k)
        print("Validation error with the former w and validation_data_classification: ",error_validation_average)
        print("------")
        error[i] = [error_train,error_validation,error_train_average,error_validation_average]
        
print("------")
print(math.ceil(len(feature_vectors)/500) - 1)


train_with_argument()
#plot


#第八问，通过最低的那个training set，的那个数值去做validation set
"""
question eight:
Experiment with various maximum iterations on the two algorithms checking performance on the validation set. 
Optionally you can try to change X from question 2. Report the best validation error for the two algorithms

"""
def train_with_different_X(X):
    # create a new vocabulary list with different X
    vocabulary_list = []
    for word in counts:
        if counts[word]>=X:
            vocabulary_list.append(word)
    # create new feature_vector with vocabulary_list
    for i in range(1,12):
        feature_vectors = []
        feature_vectors = [[1 if word in vector else 0 for word in vocabulary_list] for vector in training_set]
        w,k,iter = perceptron_train_with_argument(feature_vectors, training_data_classifications,i)
        error_train_validation = perceptron_test(w, feature_vector_validation, validation_data_classifications)
        print("with",iter)
        print("the error for validation data:", error_train_validation)
        print("---")

for X in range(30,100):
    #train_with_different_X(X)
    X += 10
        