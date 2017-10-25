'''
    Abdullah Alhassan
    Project: Twitter Airline Sentiment
    Class: Data Mining
    Date: 15/12/2016
    
'''
import math
import numpy

def probabilty(index, item, class_val, table, class_index):
    count = 0
    occur = 0
    
    for row in table:
        if(class_val == row[class_index]):
            count+=1

    if(count == 0):
        return 0
    
    for row in table:
        if(row[index] == item and class_val == row[class_index]):
            occur+=1

    return float(occur)/count

def Naive_Bays_classifier(inst, table, classes, class_index ,total, show_result = False, str_result = False):
    probabilities = []
    c_index = len(inst)-1
    for class_val in classes:
        porbability_table = []
        for i in range(len(inst)-1):
            porbability_table.append(probabilty(i, inst[i], class_val, table,class_index))
        
        probability_val = 1.0
        for item in porbability_table:
            probability_val *= item
        
        count = 0
        for row in table:
            if(inst[class_index] == row[class_index]):
                count+=1
        probability_class = float(count)/len(table)
        
        NB_probablity = probability_val * probability_class
        probabilities.append(NB_probablity)

    highes_prob = max(probabilities)
    index = probabilities.index(highes_prob)
    if(show_result):
        print 'instance: ',
        print inst[:-1]
        print 'class: %d, actual: %d' %(classes[index], inst[class_index])
    elif(str_result):
        print 'instance: ',
        print inst[:-1]
        print 'class: %s, actual: %s' %(classes[index], inst[class_index])
    actual = inst[class_index]
    predicted = classes[index]
    total[classes.index(actual)][classes.index(predicted)]+=1

