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
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import numpy

# This function takes in a multidimensional list and an index
# to return a column of that list
def get_column(table,index):
    val = []
    for row in table:
        val.append(row[index])
    return val

# This frequency function will create plots depending on the
# type argument:
# type = 1 -> bar chart
# type = 2 -> pie chart
def frequency(table, index, title, typed):
    pyplot.figure()
    
    Dic = {}
    ys = []
    xs = []
    counts = get_column(table,index)
    
    for item in counts:
        if(Dic.has_key(item)):
            Dic[item]+=1
        else:
            Dic[item] = 1
    
    for item in Dic:
        ys.append(Dic[item])
        xs.append(item)
    
    # bar chart
    if(typed == 1):
        xrng = numpy.arange(len(xs))
        yrng = numpy.arange(0, max(ys) * 1.1, max(ys) * 0.2)
        pyplot.bar(xrng, ys, 0.45, align='center')
        pyplot.grid(True)
        pyplot.xticks(xrng, xs, fontsize=9)
        pyplot.yticks(yrng)
        pyplot.title('The number of tweets by the ' + title)
        pyplot.xlabel(title)
        pyplot.ylabel('counts')
        pyplot.savefig('frequency-bar-%s.pdf'%title)
        pyplot.ylim([math.ceil(min(ys)-0.5*(max(ys)-min(ys))), math.ceil(max(ys)+0.5*(max(ys)-min(ys)))])
    
    # pie chart
    elif(typed == 2):
        pyplot.figure(figsize=(8,8))
        pyplot.pie(ys, labels=xs, autopct='%1.1f%%',colors=('b', 'g', 'r', 'c', 'm', 'y', '#03A9F4', 'w', '#F1453D', '#4A148C') )
        pyplot.title('The number of tweets by the number of ' + title)
        pyplot.savefig('frequency-pie-%s.pdf'%title)
    
    pyplot.close()

def multi_freq(table,x_index,x_indexSub,y_index,title):
    pyplot.figure()
    fig, ax = pyplot.subplots()
    
    x_index_set = list(set(get_column(table,x_index)))
    x_indexSub_set = list(set(get_column(table,x_indexSub)))
    
    xGroup = group_by(table,x_index)
    
    ys = []
    for i in range(len(x_index_set)):
        ys.append(get_column(xGroup[i],y_index))
    
    originSort = []
    for j in range(len(x_index_set)):
        originSort.append([])
        for i in range(len(x_indexSub_set)):
            originSort[j].append([])


    for j in range(len(x_index_set)):
        for item in ys[j]:
            originSort[j][group_value(item,table,y_index,x_indexSub,x_indexSub_set)].append(item)

    colors = ['b', 'g', 'r', 'c', 'm', 'y','k']
    plots = ['','','','','','','']
    for j in range(len(x_index_set)):
        for i in range(len(x_indexSub_set)):
            plots[i] = ax.bar(j+(0.13*i), len(originSort[j][i]), 0.13, color=colors[i])

    
    tmp = x_index_set[:]
    for i in range(len(tmp)):
        tmp[i] = x_index_set.index(tmp[i]) + 0.26

    ax.set_xticks(tmp)
    
    for i in range(len(x_index_set)):
        x_index_set[i] = str(x_index_set[i])

    ax.set_xticklabels(x_index_set,fontsize=9)
    
    ax.legend(plots,x_indexSub_set,'best')
    pyplot.title('Total Number of tweets by ' + title )
    pyplot.xlabel('Airline')
    pyplot.ylabel('Count')
    pyplot.savefig('multi-freq-tweets-by-%s-and-airlines.pdf' %title)
    
    pyplot.close()

def group_by(table, att_index):
    # create unique list of grouping values
    grouping_values = list(set(get_column(table, att_index)))
    result = []
    # create list of n empty partitions
    for val in grouping_values:
        result.append([])
    # add rows to each partition
    for row in table:
        if(row[att_index] != 'NA'):
            result[grouping_values.index(row[att_index])].append(row[:])

    return result

def group_value(item,table,index, r_index, r_index_set):
    for row in table:
        if(item == row[index]):
            return r_index_set.index(row[r_index])#-1

def scatter(table, class_index, att_index, title_x, title_y):
    xs_list = get_column(table,class_index)
    ys_list = get_column(table,att_index)
    
    ys_categorized = list(set(ys_list))
    xs_categorized = list(set(xs_list))
    
    for i in range(len(ys_list)):
        ys_list[i] = 1+ys_categorized.index(ys_list[i])
        xs_list[i] = 1+xs_categorized.index(xs_list[i])

    xrng = numpy.arange(len(xs_categorized))
    yrng = numpy.arange(len(ys_categorized)) #numpy.arange(0, max(ys_categorized) * 1.1, max(ys_categorized) * 0.2)
    xrng = [i+1 for i in xrng]
    yrng = [i+1 for i in yrng]
    pyplot.figure()
    pyplot.plot(xs_list,ys_list,'b.')
    pyplot.grid(True)
    pyplot.xticks(xrng, xs_categorized, fontsize=9)
    pyplot.yticks(yrng, ys_categorized, fontsize=9)
    pyplot.ylim(ymin = 0, ymax = max(ys_list)+1)
    pyplot.xlim(xmin = 0, xmax = max(xs_list)+1)
    pyplot.xlabel(title_x)
    pyplot.ylabel(title_y)
    
    pyplot.title('%s vs %s'%(title_x,title_y) )
    pyplot.savefig('linear_plot %s-vs-%s.pdf'%(title_x,title_y))
    pyplot.close()

    linear_regression(xs_list,ys_list,xs_categorized,ys_categorized,title_y, title_x)


def linear_regression(xs,ys,xs_categorized,ys_categorized,ylabel,title):
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

    xrng = numpy.arange(len(xs_categorized))
    yrng = numpy.arange(len(ys_categorized)) #numpy.arange(0, max(ys_categorized) * 1.1, max(ys_categorized) * 0.2)
    xrng = [i+1 for i in xrng]
    yrng = [i+1 for i in yrng]
        
    # Plot the scatter plot
    pyplot.figure()
    p1 = pyplot.plot(xs,ys,'b.')
    pyplot.grid(True)
    pyplot.xticks(xrng, xs_categorized, fontsize=9)
    pyplot.yticks(yrng, ys_categorized, fontsize=9)
    pyplot.ylim(ymin = 0, ymax = max(ys)+1)
    pyplot.xlim(xmin = 0, xmax = max(xs)+1)
    pyplot.xlabel(title)
    pyplot.ylabel(ylabel)
    pyplot.title('%s vs %s'%(title,ylabel))
    
    # Now add the linear regression y = mx + b
    linear = [(m*xs[i] + b) for i in range(len(xs))]
    
    p2 = pyplot.plot(xs,linear, color='r')
    
    p3 = matplotlib.patches.Rectangle((0, 0), 0, 0, alpha=0.0)
    
    # Add correlation coefficient and covariance
    
    # Coveriance formula is [SUM((xs[i] - xsAvg) * (ys[i] - ysAvg))] / n
    
    total = 0
    for i in range(len(xs)):
        total += (xs[i] - xsAvg) * (ys[i] - ysAvg)

    cov = total/len(xs)

    # Correlation formula is  [n*SUM(xs*ys) - SUM(xs)*SUM(xs)] / [ sqrt((n*SUM(xs^2) - SUM(xs)^2)*(n*SUM(ys^2) - SUM(ys)^2)) ]

    corrTop1 = 0
    for i in range(len(xs)):
        corrTop1 += (xs[i] * ys[i])


    corrTop2 = 0
    for i in range(len(ys)):
        corrTop2 += (ys[i])

    corrTop3 = 0
    for i in range(len(xs)):
        corrTop3 += (xs[i])

    corrTop = (len(xs)*corrTop1) - (corrTop2 * corrTop3)
    
    corrBot1 = 0
    for i in range(len(xs)):
        corrBot1 += (xs[i] * xs[i])

    corrBot2 = 0
    for i in range(len(xs)):
        corrBot2 += (xs[i])
    corrBot2 *= corrBot2
    
    corrBot3 = 0
    for i in range(len(ys)):
        corrBot3 += (ys[i] * ys[i])
    
    corrBot4 = 0
    for i in range(len(ys)):
        corrBot4 += (ys[i])
    corrBot4 *= corrBot4
    
    corrBot = math.sqrt((len(xs)*corrBot1 - corrBot2) * (len(ys)*corrBot3 - corrBot4))
    
    corr = corrTop/corrBot
    l = pyplot.legend([p3], ['corr: %0.2f, cov: %0.2f'%(corr,cov)] , 'upper right', handlelength=0 )
    for text in l.get_texts():
        text.set_color("red")

    pyplot.savefig('linear_regression-%s-vs-%s.pdf'%(title,ylabel))
    pyplot.close()





