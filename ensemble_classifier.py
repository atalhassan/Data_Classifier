'''
    Abdullah Alhassan
    Project: Twitter Airline Sentiment
    Class: Data Mining
    Date: 15/12/2016
    
'''
import math
import operator
import os
import sys
import numpy
from random import randint
import random
from math import log

# Ensemble random forest taking all neccessary attributes
# to generate classifiers
def random_forest(table, class_index, N, M, F, k):
    remainder_set, test_set = holdout_partition(table)
    DTs = []
    validation_sets = []
    M_best_DT = []
    Accuracies = []
    
    class_domain = list(set(get_column(table,class_index)))
    
    for i in range(N):
        # randomize remainder_set
        dataset = bootstrap(remainder_set)
        training, validation = holdout_partition(dataset)
        att_indexes = []
        for index in range(len(table[0])):
            if(index != class_index):
                att_indexes.append(index)
        att_domains = []
        for index in att_indexes:
            att_domains.append(list(set(get_column(table,index))))
        DT = tdidt(training, att_indexes, att_domains, class_index, F)
        DTs.append(DT)
        
        # Validate DT
        total = []
        for item in class_domain:
            total.append([0]*len(class_domain))

        D = stratified(validation,k)
        for i in range(0,k):
            test = D[i]
            for item in test:
                label = tdidt_classifier(DT, item, class_domain)
                actual = class_domain.index(item[class_index])
                predicted = class_domain.index(label)
                total[actual][predicted] += 1

        acc = accuracy(total)
        Accuracies.append(acc)

    # select M accurate trees
    Accuracies_copy = Accuracies[:]
    Accuracies_copy.sort()
    ensemble_acc = []
    for acc in Accuracies_copy[-M:]:
        DT_index = Accuracies.index(acc)
        ensemble_acc.append(acc)
        M_best_DT.append(DTs[DT_index])

    return M_best_DT, test_set, remainder_set


def bootstrap(table):
    return [table[randint(0,len(table)-1)] for _ in table]

# Top down Induction Decision Tree
def tdidt(instances, att_indexes, att_domains, class_index, F = 0):
    if(same_class(instances, class_index)):
        stats = ['Leaves']
        stats.append(partition_stats(instances , class_index))
        decision_tree = stats
        return decision_tree
    
    if(not att_indexes):
        stats = ['Leaves']
        stats.append(partition_stats(instances , class_index))
        decision_tree = stats
        return decision_tree
    
    if(not instances):
        stats = ['Leaves']
        stats.append(partition_stats(instances , class_index))
        decision_tree = stats
        return decision_tree
    
    selected_index = select_attribute(instances, att_indexes, class_index, F)
    
    decision_tree = ['Attribute']
    decision_tree.append(att_indexes[selected_index])

    value = partition_instances(instances, att_indexes[selected_index] ,selected_index, att_domains)
    
    del att_indexes[selected_index]
    del att_domains[selected_index]
    for key in value:
        part = ['Value', key]
        part.append(tdidt( value[key] , att_indexes[:], att_domains[:], class_index, F))
        decision_tree.append(part)

    return decision_tree

def partition_instances(instances, att_index, selected_index, att_domains):
    partition = {}
    for val in att_domains[selected_index]:
        partition[val] = []
    for row in instances:
        for val in att_domains[selected_index]:
            if(row[att_index] == val):
                partition[val].append(row)
                break
    return partition

def same_class(instances, class_index):
    return (len(list(set(get_column(instances,class_index)))) == 1)

# returns [[label1, occ1, tot1], [label2, occ2, tot2], ...]
def partition_stats(instances, class_index):
    class_domain = list(set(get_column(instances,class_index)))
    stats = []
    for item in class_domain:
        stats.append([item,0,0])
    
    total = 0
    
    for row in instances:
        index = class_domain.index(row[class_index])
        total += 1
        stats[index][1] += 1
    
    for i in range(len(class_domain)):
        stats[i][2] = total
    
    return stats

def select_attribute(instances, att_indexes, class_index, F):
    # select F random att
    if( (len(att_indexes) > F) and (F != 0) ):
        shuffled = att_indexes[:]  # make a copy
        random.shuffle(shuffled)
        attributes = shuffled[:F]
    else:
        attributes = att_indexes
    
    E_new = []
    for index in attributes:
        E_new.append(calc_enew(instances, index, class_index))
    
    # find the min Entrpy to partition on
    minVal = min(E_new)
    partitionIndex = E_new.index(minVal)
    return partitionIndex


def calc_enew(instances, att_index, class_index):
    # get the length of the partition
    D = len(instances)
    # calculate the partition stats for att_index (see below)
    freqs = attribute_frequencies(instances, att_index, class_index)
    # find E_new from freqs (calc weighted avg)
    E_new = 0
    for att_val in freqs:
        D_j = float(freqs[att_val][1])
        probs = [(c/D_j) for (_, c) in freqs[att_val][0].items()]
        if 0 in probs:
            E_D_j = 0
        else:
            E_D_j = -sum([p*log(p,2) for p in probs])
        E_new += (D_j/D)*E_D_j
    return E_new

def attribute_frequencies(instances, att_index, class_index):
    # get unique list of attribute and class values
    att_vals = list(set(get_column(instances, att_index)))
    class_vals = list(set(get_column(instances, class_index)))
    # initialize the result
    result = {v: [{c: 0 for c in class_vals}, 0] for v in att_vals} # build up the frequencies
    for row in instances:
        label = row[class_index]
        att_val = row[att_index]
        result[att_val][0][label] += 1
        result[att_val][1] += 1
    return result


def tdidt_classifier(DT, instance, class_domain):
    # Base Case
    label = class_domain[0]
    if(DT[0] == 'Attribute'):
        index = DT[1]
        for value in DT:
            if type(value) is list:
                if(value[0] == 'Value'):
                    val = value[1]
                    if(instance[index] == val):
                        label = tdidt_classifier(value[2], instance, class_domain)
                        
    elif(DT[0] == 'Leaves'):
    # empty path
        
        if(not DT[1]):
            return class_domain[len(class_domain) - 1]
        leaves = DT[1]
        vote = {}
        for label in leaves:
            vote[label[0]] = label[1]
        return max(vote, key=vote.get)
    return label

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

# This function takes in a multidimensional list and an index
# to return a column of that list
def get_column(table,index):
    val = []
    for row in table:
        val.append(row[index])
    return val

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



