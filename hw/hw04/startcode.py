# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:58:31 2019

@author: ruby_
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import cross_validation, svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.multiclass import OneVsRestClassifier

import matplotlib.pyplot as plt
import numpy as np
X = np.array([(2,2), (4,4), (4,0), (0,0), (2,0), (0,2)])
labels = np.array([1,1,1,-1,-1,-1])

positive = np.where(labels==1)
negative = np.where(labels==-1)

x_positive = X[positive]
x_negative = X[negative]
print(x_positive)
for i in range(len(x_positive)):
    plt.scatter(x_positive[i,0],x_positive[i,1], color = 'red',marker = "*")
for i in range(len(x_negative)):
    plt.scatter(x_negative[i,0],x_negative[i,1], color = 'orange',marker = "o")
plt.xlabel('x_1')
plt.ylabel('x_2')

x = np.arange(0, 4, 0.1)
y = -x +3
plt.plot(x, y)

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
    with open("data/spam_train.txt") as training_data_file:
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
feature_vectors.pop(0)
feature_vectors = np.array(feature_vectors)
print(type(feature_vectors))


def pegasos_svm_train(feature_vectors,training_data_classifications):
    lambda_ = 2**-5
    num_iter = 20
    num_features = len(feature_vectors[1])
    num_example = len(feature_vectors[0])
    w = np.zeros((num_features,),dtype='int64')
    obj = []
    hinge = 0
    t = 0
    for i in range(num_iter):
        obj_sum = 0
        for j in range(num_example):
            t+=1
            neta = 1/(t*lambda_)
            x_j = feature_vectors[j]
            y_j = 1
            if training_data_classifications[j] == 0:
                y_j = -1
            obj_sum += max(0, 1-(y_j*np.dot(w,x_j)))
            if y_j * np.dot(x_j,w) < 1:
                w = (1-neta*lambda_)*w + neta*y_j*x_j
            else:
                w = (1-neta*lambda_)*w
        obj.append(((lambda_/2)*(np.dot(w, w))) + ((obj_sum+max(0, 1-(training_data_classifications[num_examples-1] *np.dot(w, feature_vectors[num_example-1]))))/num_example))
    return w

w = pegasos_svm_train(feature_vectors,training_data_classifications)
            
def pegasos_svm_train(data,label,myLambda):
    t = 0
    numMistakes = 0
    totalHingeLoss = 0
    maxPasses = 20
    featureVectorLength = len(data[1])
    svmObjective = []
    w = np.zeros(featureVectorLength)
    for i in range(maxPasses):
        for j in range(len(data)):
            t += 1
            eta = 1/(t*myLambda)
            prediction = int(label[j]) * np.dot(w, data[j])
            #use support vectors
            if (prediction) < 1:
                #print((1-(eta*myLambda)) * w)
                w = ((1-(eta*myLambda)) * w) + (eta *int(label[j])*data[j])
            else:
                w = ((1-(eta*myLambda)) * w)
            #use classifier
            if (prediction) < 0:
                numMistakes +=1
        hingeLoss = 0
        for k in range(len(data)):
            hingeLoss += max(0, 1 - int(label[i])*np.dot(data[i], w))
        hingeLoss = hingeLoss / len(label)
        totalHingeLoss += hingeLoss
        svmObjective.append((t, ((myLambda/2) * math.pow(np.linalg.norm(w), 2)) + hingeLoss))
    return (w, svmObjective, numMistakes/(maxPasses*len(data)), totalHingeLoss/maxPasses)


pegasos_svm_train(feature_vectors,training_data_classifications,2**-5)

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
    




















 