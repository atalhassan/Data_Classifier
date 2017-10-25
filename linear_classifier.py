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


# This function will calculate the linear regression and
# predict the y values based on " y = mx + b" and it will
# calculate the accuracey and precision of the classification
def linear_regression_classifier(table, index_x ,index_y):
    xs_list = get_column(table,index_x)
    ys_list = get_column(table,index_y)
    
    global ys_categorized
    ys_categorized = list(set(ys_list))
    global xs_categorized
    xs_categorized = list(set(xs_list))
        
    for i in range(len(ys_list)):
        ys_list[i] = ys_categorized.index(ys_list[i])
        xs_list[i] = xs_categorized.index(xs_list[i])
    
    xs = xs_list
    ys = ys_list
    
    # First we have to find m in "y = mx + b"
    xsAvg = numpy.average(xs)
    ysAvg = numpy.average(ys)
    
    top = []
    for i in range(len(xs)):
        top.append((xs[i] - xsAvg) * (ys[i] - ysAvg))
    
    bot = []
    for i in range(len(xs)):
        bot.append((xs[i] - xsAvg)*(xs[i] - xsAvg))

    m = sum(top)/sum(bot)

    # Second find b which is..
    b = ysAvg - m * xsAvg
    return [m,b]
    
def classify(cutoff_classes,inst,index_x,total,rule):

    cutoffs = cutoff_classes[:]
    for i in range(len(cutoffs)):
        cutoffs[i] = xs_categorized.index(cutoffs[i])

    instance = inst[:]
    for i in xs_categorized:
        if(i == instance[index_x]):
            instance[index_x] = xs_categorized.index(i)
            break

        
    x = instance[index_x]
    
    y = (rule[0]*int(x)) +rule[1]
    actual_y = instance[index_x]
    
    foundPredict = False
    foundActual = False
    
    predict = len(cutoffs)-1
    actual = len(cutoffs)-1
    
    for i in range(len(cutoffs)):
        if( (y <= cutoffs[i]) and not(foundPredict)):
            predict = i
            foundPredict = True
        if( (actual_y <= cutoffs[i]) and not(foundActual)):
            actual = i
            foundActual = True

    
    total[actual][predict] += 1

