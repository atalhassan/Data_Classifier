'''
Abdullah Alhassan
Project: Twitter Airline Sentiment
Class: Data Mining
Date: 15/12/2016

Description:
    In this project there will be five types of classifiers ensemble, decision tree,
    k-nearest neighbor, linear regression, and naive bays. All of them will be used on
    a dataset called twitter.txt that conatins the follwing attributes:
    
    Atributes:
        0 : Airline
        1 : Retweet Count
        2 : Day of tweet
        3 : Sentiment
        
    The class attribute that the classifiers will pridect is the sentiment.

'''

import math
import operator
import os
import csv
import sys
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import numpy
from tabulate import tabulate
from random import randint
import random
from math import log
import ensemble_classifier
import KNN_Classifier
import Visualization
import linear_classifier
import Naive_Bays
from multiprocessing import Process


def holdout_partition(table):
    # randomize the table
    randomized = table[:]
    n = len(table)
    for i in range(n):
        # pick an index to swap
        j = randint(0, n-1)
        randomized[i], randomized[j] = randomized[j], randomized[i]
    # return train and test sets
    n0 = (n*2)/3
    return randomized[0:n0], randomized[n0:]


def stratified(table, k):
    index = range(len(table))
    for i in range(len(table)):
        rand = randint(0, len(table)-1)
        index[i] , index[rand] = index[rand] , index[i]
    newTable = []
    for i in range(k):
        newTable.append([])
    
    rem = len(table) % k
    width = len(table)/k
    start = 0
    w = 0
    for i in range(k):
        w += width
        for item in index[start:w]:
            newTable[i].append(table[item])
        start += width
    
    for item in index[-rem:]:
        newTable[k-1].append(table[item])
    
    return newTable

def accuracy(total):
    sum_att = 0
    for row in total:
        for item in row:
            sum_att+=item
        row.append(sum_att)
        sum_att = 0
    
    sum_att = 0
    for i in range(len(total)):
        sum_att+=total[i][i]
    
    sum_total = 0
    for row in total:
        sum_total+=row[len(row)-1]
    
    return float(sum_att)/sum_total

# performs majority voting
def majority_voting(labels, class_domain):
    dic = {}
    for label in class_domain:
        dic[label] = 0
    
    for label in labels:
        dic[label] += 1
    
    return max(dic, key = dic.get)

# This function takes in a multidimensional list and an index
# to return a column of that list
def get_column(table,index):
    val = []
    for row in table:
        val.append(row[index])
    return val

def read_csv(filename):
    theFile = open(filename, "r")
    theReader = csv.reader(theFile, dialect='excel')
    table = []
    for row in theReader:
        if(len(row) > 0):
            table.append(row)
    theFile.close()
    return table

def remove_missing_values(table):
    for row in table[:]:
        for item in row:
            if(item == 'NA'):
                table.remove(row)
                break
    return table


##########
#  main  #
##########
def main():
    
    # retrieve the data from the file
    dataTable  = read_csv('Tweets.txt')
    lengthBefore = len(dataTable)
    dataTable = remove_missing_values(dataTable)
    lengthAfter = len(dataTable)
    
    # create handy golbal represetation of some attribute sets
    # and use its index as categorical value
    global ailines_set # contain all ariline names
    ailines_set = list(set(get_column(dataTable,0)))
    global Day_set # contain the day names
    Day_set = list(set(get_column(dataTable,2)))
    
    # Print informatin about the data
    
    print '==================================================='
    print ' Information'
    print '=================================================='
    
    print 'Length of data before removing rows with empty attributes: %d' %lengthBefore
    print 'Length of data after removing rows with empty attributes: %d' %lengthAfter
    print 'Attributes:'
    print '           Airlines        : ',
    print ailines_set
    print '           Retweet         : ',
    print list(set(get_column(dataTable,1)))
    print '           Days            : ',
    print Day_set
    print '           Sentiment       : ',
    print list(set(get_column(dataTable,3)))
    print ''
    
    # Create vizulation for the preprocessing step 
    
    Visualization.frequency(dataTable, 0, "Airlines", 1)
    Visualization.frequency(dataTable, 1, "number of Retweet", 1)
    Visualization.frequency(dataTable, 2, "Day of tweet", 1)
    Visualization.frequency(dataTable, 3, "Sentiment", 1)
    
    # multiple frequency diagram for the Airlines and days
    Visualization.multi_freq(dataTable,0,2,2,'Day of tweet')
    Visualization.multi_freq(dataTable,0,3,3,'Sentiment')

    #linear plot
    Visualization.scatter(dataTable, 3, 2, "Sentiment", "Day of tweet")
    
    print '==================================================='
    print ' Ensemble Classifier - predictive accuracy'
    print '=================================================='
    
    # Ensemble attributes
    F = 2
    N = 50
    M = 37
    k = 10
    
    print 'F = %d  ,  N = %d  ,  M = %d' %(F,N,M)
    
    class_index = 3
    class_domain = list(set(get_column(dataTable,class_index)))
    
    ensemble, test_set, remainder = ensemble_classifier.random_forest(dataTable, class_index, N, M, F, k)
    
    # Calculate accuracy
    total = []
    for item in class_domain:
        total.append([0]*len(class_domain))

    i = 0
    for test in test_set:
        labels = []
        for tree in ensemble:
            label = ensemble_classifier.tdidt_classifier(tree, test, class_domain)
            labels.append(label)
        label = ensemble_classifier.majority_voting(labels, class_domain)
        actual = class_domain.index(test[class_index])
        predicted = class_domain.index(label)
        total[actual][predicted] += 1
        
    acc = accuracy(total)
    
    print 'accuracy = %0.2f, error rate = %0.2f' %(acc, 1-acc)

    print '\n'
    print '==================================================='
    print ' Ensemble  - Confusion Matrix'
    print '=================================================='
    last = len(total[0])-1
    for i in range(len(total)):
        if(total[i][last] != 0):
            acc = float(total[i][i])/total[i][last]
        else:
            acc = 0
        total[i].append('%0.2f' %(acc * 100))

    
    for i in range(len(total)):
        total[i].insert(0,class_domain[i])
    
    headers= class_domain + ['Total', 'Recognition(%)']
    
    print tabulate(total, headers, tablefmt='orgtbl')
    print '\n'
    
    
    print '==================================================='
    print 'DT  - predictive accuracy'
    print '=================================================='
    
    att_indexes = []
    for index in range(len(remainder[0])):
        if(index != class_index):
            att_indexes.append(index)
    att_domains = []
    for index in att_indexes:
        att_domains.append(list(set(get_column(remainder,index))))

    DT = ensemble_classifier.tdidt(remainder, att_indexes, att_domains, class_index)
    
    # Calculate accuracy
    accuracy_list = []
    total = []
    for item in class_domain:
        total.append([0]*len(class_domain))

    D = stratified(test_set,k)
    for i in range(0,k):
        test = D[i]
        for item in test:
            label = ensemble_classifier.tdidt_classifier(DT, item, class_domain)
            actual = class_domain.index(item[len(item)-1])
            predicted = class_domain.index(label)
            total[actual][predicted] += 1

    acc = accuracy(total)
    
    print 'accuracy = %0.2f, error rate = %0.2f' %(acc, 1-acc)
    
    print '\n'
    print '==================================================='
    print 'DT  - Confusion Matrix'
    print '=================================================='
    
    last = len(total[0])-1
    for i in range(len(total)):
        if(total[i][last] != 0):
            acc = float(total[i][i])/total[i][last]
        else:
            acc = 0
        total[i].append('%0.2f' %(acc * 100))


    for i in range(len(total)):
        total[i].insert(0,class_domain[i])
    
    headers= class_domain + ['Total', 'Recognition(%)']
    print tabulate(total, headers, tablefmt='orgtbl')
    print '\n'
    
    print '\n'
    print '==================================================='
    print 'K-NN_Classifier - Predictive Accuracy'
    print '=================================================='
    print 'k = %d' %(k)

    att_indexs = []
    for i in range(len(dataTable[0])):
        if(i != class_index):
            att_indexs.append(i)

    
    # K-NN
    total = []
    for item in class_domain:
        total.append([0]*len(class_domain))
    
    top_k_rows = []
    training_set, test = holdout_partition(dataTable)

    for t in test[0:500]:
        top_k_rows = KNN_Classifier.k_NN_classifier(training_set, att_indexs, t, k, class_domain, class_index, total)
        KNN_Classifier.select_class_label(top_k_rows, t, class_domain, class_index, total) 
        

    acc = accuracy(total)
    print 'k Nearest Neighbors: accuracy = %0.2f, error rate = %0.2f'%(acc,1.0-acc)

    print '\n'
    print '==================================================='
    print 'K-NN  - Confusion Matrix'
    print '=================================================='

    last = len(total[0])-1
    for i in range(len(total)):
        if(total[i][last] != 0):
            acc = float(total[i][i])/total[i][last]
        else:
            acc = 0
        total[i].append('%0.2f' %(acc * 100))

        
    for i in range(len(total)):
        total[i].insert(0,class_domain[i])
    
    headers= class_domain + ['Total', 'Recognition(%)']

    print tabulate(total, headers, tablefmt='orgtbl')
    print '\n'
    
    
    print '\n'
    print '==================================================='
    print 'linear regression classifier - Predictive Accuracy'
    print '=================================================='
    
    total = []
    for item in class_domain:
        total.append([0]*len(class_domain))
    
    # Linear Regression
    training_set, test = holdout_partition(dataTable)
    rule = linear_classifier.linear_regression_classifier(training_set, class_index, 2)
    for item in test:
        linear_classifier.classify(class_domain, item, class_index,total,rule)
        
        
    acc = accuracy(total)
    
    print 'Linear Regression: accuracy = %0.2f, error rate = %0.2f'%(acc,1.0-acc)

    
    print '\n'
    print '==================================================='
    print 'Linear Regression - Confusion Matrix'
    print '=================================================='

    last = len(total[0])-1
    for i in range(len(total)):
        if(total[i][last] != 0):
            acc = float(total[i][i])/total[i][last]
        else:
            acc = 0
        total[i].append('%0.2f' %(acc * 100))

        
    for i in range(len(total)):
        total[i].insert(0,class_domain[i])
    
    headers= class_domain + ['Total', 'Recognition(%)']

    print tabulate(total, headers, tablefmt='orgtbl')
    print '\n'
    
    
    print ''
    print '==========================================='
    print 'Naive Bays: Predictive Accuracy'
    print '==========================================='
    
    total = []
    for item in class_domain:
        total.append([0]*len(class_domain))


    D = stratified(dataTable,10)
    for i in range(k):
        test = D[i]
        training_set = []
        for j in range(k):
            if(j != i):
                training_set += D[j]
        for item in test[0:50]:
            Naive_Bays.Naive_Bays_classifier(item, training_set, class_domain, class_index,total)

    acc = accuracy(total)
    print 'accuracy = %0.2f, error rate = %0.2f' %(acc, 1-acc)
    print ''

    print '\n'
    print '==================================================='
    print 'Naive Bays - Confusion Matrix'
    print '=================================================='

    last = len(total[0])-1
    for i in range(len(total)):
        if(total[i][last] != 0):
            acc = float(total[i][i])/total[i][last]
        else:
            acc = 0
        total[i].append('%0.2f' %(acc * 100))

        
    for i in range(len(total)):
        total[i].insert(0,class_domain[i])
    
    headers= class_domain + ['Total', 'Recognition(%)']

    print tabulate(total, headers, tablefmt='orgtbl')
    print '\n'

    
if __name__ == '__main__':
    main()
