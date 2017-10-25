'''
    Abdullah Alhassan
    Project: Twitter Airline Sentiment
    Class: Data Mining
    Date: 15/12/2016
    
    
    '''
import math
import operator
import os
import csv
import sys
import numpy


# This function takes in a multidimensional list and an index
# to return a column of that list
def get_column(table,index):
    val = []
    for row in table:
        val.append(row[index])
    return val

def k_NN_classifier(training_set, n_att_list, instance, k, cutoffs, index, total = []):
    row_distances = []
    for row in training_set:
        d = distance(row, instance, n_att_list)
        row_distances.append([d, row])
    top_k_rows = get_top_k(row_distances, k)
    return top_k_rows
    #select_class_label(top_k_rows, instance, cutoffs, index, total)
    
def select_class_label(top_k, instance, cutoffs, index, total = []):
    vote = {}
    for item in cutoffs:
        vote[item] = 0
    
    for row in top_k:
        vote[row[index]] += 1
    
    predict = max(vote, key=vote.get)

    actual = instance[index] #len(cutoffs)

    if(total):
        predict_index = cutoffs.index(predict)
        actual_index = cutoffs.index(actual)
        total[actual_index][predict_index] += 1
    else:
        print 'Class: %d, ' %(predict),
        print 'Actual: %d' %(actual)

def get_top_k(row_distances, k):
    col = get_column(row_distances,0)
    sortedCol = col[:]
    sortedCol.sort()
    col2 = get_column(row_distances,1)
    top = []
    for i in range(k):
        index = col.index(sortedCol[i])
        top.append(col2[index])
    return top

def distance(row, instance, n_att_list):
    d = 0
    for index in n_att_list:
        if(row[index] == instance[index]):
            d += 1
            #d += ((float(row[index]) - float(instance[index]))**2)
    return d#math.sqrt(d)

