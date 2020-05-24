#!/usr/bin/env python
# coding: utf-8

# <a name='home'></a>
# # Table of Content
# 1. [Modul 1:](#mod1)  
#     1.1 [Simple Regression: Motivation](#SRmot)  
#     1.2 [Simple Regression: Representation](#SRpre)  
#     1.3 [Simple Regression: Estimation](#SRest)  
#     1.4 [Simple Regression: Evaluation](#SReva)  
#     1.2 [Simple Regression: Application](#SRapp) 
# 2. [Modul 2:](#mod2)
# 3. [Modul 3:](#mod3)
# 4. [Modul 4:](#mod4)
# 5. [Modul 5:](#mod5)
# 6. [Modul 6:](#mod6)
# 7. [Modul 7:](#mod7)
# 8. [Modul 8:](#mod8)
#  
# A. [Training Question](#train)  
#     A.a [Solution Training Exercise 1.1](#train1_1)
# 
# ---
# 

# <a name='mod1'></a>
# ## Modul 1:
# <a name='SRmot'></a>
# ### Simple Regression: Motivation
# 
# Define the turnover as product of price and sales, where Sales = a + b x P with a>0 and b<0. Derive the formula for the optimal price in terms of and b.
# 
# P = Price, T = Turnover = Price x Sales  
# 
# $ T = P(a+bP) = aP + bP^2$  
# $\frac{dT}{dP} = a + 2P = 0$  
# $P_{Opt} = - \frac{a}{2b}$
# 
# In a simple regression, we focus on two variables of interest we denote by y and x, where one variable, x, is thought to be helpful to predict the other, y. This helpful variable x we call the **regressor** variable or the **explanatory factor**. And the variable y that we want to predict is called the **dependent** variable, or the **explained** variable $\hat{y}$.
# 
# If the histogram of our data suggest that the distribution can be approximated by a **normal distribution** a simple way of summarizing the observations is written as:  
# 
# $ S \sim $NID$(\mu,\sigma^2)$
# 
# [Slides](https://d396qusza40orc.cloudfront.net/eureconometrics-assets/Dataset%20Files%20for%20On-Demand%20Course/Exercises%20and%20datasets/Handouts%20slides%20videos/Lecture%201.1-4on1.pdf)
# 
# [Home](#home)  
# ___

# <a name='train1_1'></a>
# ## Solution Training Exercise 1.1
# **Questions**
# 1. Make two histograms, one of expenditures and the other of age. Make also a scatter diagram with expenditures on the vertical axis versus age on the horizontal axis.
# 2. In what respect do the data in this scatter diagram look different from the case of the sales and price data discussed in the lecture?
# 3. Propose a method to analyze these data in a way that assists the travel agent in making recommendations to future clients.
# 4. Compute the sample mean of expenditures of all 26 clients.  
# 5. Compute two sample means of expenditures, one for clients of age forty or more and the other for clients of age below forty.  
# 6. What daily expenditures would you predict for a new client of fifty years old? And for someone who is twenty-five years old? 
#     
# ### Question 1
# Make two histograms, one of expenditures and the other of age. Make also a scatter diagram with expenditures on the vertical axis versus age on the horizontal axis.

# In[1]:


#Import the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load the dataset to a datafram (df) and have a glance at the 5 first records of the dataset
df = pd.read_csv('TrainExer11.csv')
df.head()


# In[3]:


#Define the bin size for the histogram's
df['Expenditures'].head()
count, bin_edges = np.histogram(df['Expenditures'])


# In[4]:


#Plot the histogram for the expenditure
df['Expenditures'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges, edgecolor='white')
plt.title('Histogram of Expenditures')
plt.ylabel('Number of Observations')
plt.xlabel('Size of Expenditures')
plt.show()


# In[5]:


#Define the bin size for the histogram's
df['Age'].head()
count, bin_edges = np.histogram(df['Age'])


# In[6]:


#Plot the histogram for the age
df['Age'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges, edgecolor='white')
plt.title('Histogram of Age')
plt.ylabel('Number of Observations')
plt.xlabel('Age of participants')
plt.show()


# In[7]:


#Plot a scatter diagram with Age on the horizontal axis and expenditure on the vertical axis
df.plot(kind='scatter', x='Age', y='Expenditures', figsize=(10, 6), color='darkblue')
plt.title('Total Expenditure based on Age')
plt.xlabel('Age')
plt.ylabel('Expenditure')
plt.show()


# [Home](#home)  
# ___

# ### Question 2  
# 
# In what respect do the data in this scatter diagram look different from the case of the sales and price data discussed in the lecture?
# 
# 1. The histograms do not support the assumption of a normal distribution: Not over the whole dataset, neither within the two cluster of expenditure
# 2. The scatter diagram does not depict a clear linear regression of the data points. At best it shows a linear regression with a weak correlation, i.e. large residual (error) values
# 
# [Home](#home)  
# ___
# 
# ### Question 3
# Propose a method to analyze these data in a way that assists the travel agent in making recommendations to future clients.
# 
# Based on the scatter plot, there are two distinct groups of customer:
# * Group 1 is below 40 years of age with an expenditure >103 and <110
# * Group 2 is above 40 years of age with an expenditure >90 and <103  
# 
# Hence an offering for each of the groups' expenditure habit, will meet the clusters specific needs better.
# 
# [Home](#home)  
# ___

# ### Question 4
# Compute the sample mean of expenditures of all 26 clients.
# 
# The **sample mean** or **empirical mean** is $ \frac{1}{N}\sum_{i=1}^{N}{x_i}$

# In[8]:


df_mean=df['Expenditures'].mean()
print("The sample mean of expenditures of all 26 clients is: "+"{:.2f}".format(df_mean));


# [Home](#home)  
# ___
# ### Question 5
# Compute two sample means of expenditures, one for clients of age forty or more and the other for clients of age below forty.
# 

# In[9]:


df_grp1 = df[df['Age']<39.9]
df_grp2 = df[df['Age']>39.9]
df_mean1 = df_grp1['Expenditures'].mean()
df_mean2 = df_grp2['Expenditures'].mean()
print("The sample mean of expenditures of the group below 40 is: "+"{:.2f}".format(df_mean1),
     "\nThe sample mean of expenditures of the group above or equal to 40 is: "+"{:.2f}".format(df_mean2))


# [Home](#home)  
# ___
# ### Question 6
# What daily expenditures would you predict for a new client of **fifty** years old? And for someone who is **twenty-five** years old?
# 

# In[10]:


from sklearn import linear_model
import numpy as np
import seaborn as sns
#import plotly.graph_objects as go


# In[11]:


x1 = df_grp1['Age']      # year on x-axis
y1 = df_grp1['Expenditures']     # total on y-axis
fit1 = np.polyfit(x1, y1, deg=1)

x2 = df_grp2['Age']      # year on x-axis
y2 = df_grp2['Expenditures']     # total on y-axis
fit2 = np.polyfit(x2, y2, deg=1)

print('Group below 40 years of age:')
print('Coefficient of the group below 40 is: '+'{:.4f}'.format(fit1[0]), '\nThe intercept of the group below 40 is: '+'{:.4f}'.format(fit1[1]))
print('\nGroup of 40 years of age or older:')
print('Coefficient of the group equal to or older than 40 is: '+'{:.4f}'.format(fit2[0]), '\nThe intercept of the group equal to or older than 40 is: '+'{:.4f}'.format(fit2[1]))


# In[12]:


#Representation of the linear regression per sub-group
plt.figure(figsize=(15, 10))

ax1 = sns.regplot(x='Age', y='Expenditures', data=df_grp1, ci=None, marker='+')
ax2 = sns.regplot(x='Age', y='Expenditures', data=df_grp2, ci=None, marker='+')

ax1.set(xlabel='Age', ylabel='Expenditures') # add x- and y-labels
ax1.set_title('Linear regression for the two sub-groups in one chart') # add title

plt.show()


# In[13]:


y1_hat = fit1[0]*25+fit1[1]
y2_hat = fit2[0]*50+fit2[1]
print('The estimated expenditure for customer of 25 years of age is: '+'{:.2f}'.format(y1_hat), '\nand the estimated expenditure for a customer of 50 years of ages is: '+'{:.2f}'.format(y2_hat))


# [Home](#home)  
# ___

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




