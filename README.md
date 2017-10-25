
# Airline Sentiment from Twitter Feeds

### Introduction

A 14640 twitter feeds were collected during February 2015. The feeds contained an airline mention, specifically these airlines: United, Virgin America, American, Delta, US Airways, and Southwest. Those tweets, then, were rated into three categories regarding a feedback: positive, neutral, and negative. Originally, the dataset contains 13 attributes shown in Table 1, then data cleaning has been performed to on the dataset to make it useful. A major cleaning is removing the key attributes such as tweet id, and other attribute unnecessary attribute that does not help classify an airline sentiment such as name, tweet\_coor, and user\_timezone. The only attributes left that help generate useful classifiers are airline, retweet\_count, tweet\_created, and airline\_sentiment. The tweet\_created attribute was given in date-time format e.g &quot;YYYY/MM/DD HH:MM&quot;, which was converted to a day of week name that will help categorizes airline performance according to the day of week. In this project there were five types of classifiers ensemble, decision tree, k-nearest neighbor, linear regression, and naive bays.



| **Attribute** | **Explanation** |
| --- | --- |
| tweet \_id | Is the unique key for every tweet |
| negativereason | Negative comments of the flight |
| airline | The name of the airline |
| airline\_sentiment\_gold | Is the result after testing the dataset on a classifier |
| name | The name of person tweeting |
| negativereason\_gold | Is the result after testing the dataset on a classifier |
| retweet\_count | Number of retweets |
| text | The actual tweet |
| tweet\_coord | The coordinates of the tweet |
| tweet\_created | The date of when the tweet was created |
| tweet\_location | The location of the tweet |
| user\_timezone | The time zone of the user |
| airline\_sentiment | The class label: positive, neutral, and negative |

Table 1: Dataset Attributes

### Data Analysis Techniques

* ensemble_classifier
* KNN_Classifier
* linear_classifier
* Naive_Bays

