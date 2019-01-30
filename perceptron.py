
# -*- coding: utf-8 -*-
"""
Created on Sun 1:16 AM 03/02/2018

@author: SELVA GANESH
"""
#==========================================================#
# 	PERCEPTRON - IMPLEMENTATION		   #
#==========================================================#


#Import libraries

from __future__ import division
import sys as sy
import math 
import os
import random as r

from decimal import *

from nltk.stem import *

#==========================================================#

#Declare global variables

tot_words = []	#All the words in spam/ham files
tot_words_cnt = 0

w_random = []
train_file_list = []
total_list = []
w_vec = []

spam_id = 1
ham_id  = 0
one_cnst = 1

data_dict = { }

dw = []

w_list = [ 'w_val' ]
c_list = [ 'class' ]

features  = []	#Unique words in the overall dataset
features_cnt = 0

#Iteration for learning weight
def_step = 5

#Learning rate
neeta = 0.3


#==========================================================#

#Function assignment
stemmer = PorterStemmer()

#Get decimal precision upto 28 digits
getcontext().prec = 28	

#==========================================================#

#Special Character
spl_char = [ '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', ':', '"', '<', '>', '?', '/', ';', "'", '|', '\'', '[', ']', ',', '.']

stop_words = [ 'a' , 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having','he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours' ]


#==========================================================#
#List of Lambda functions



#==========================================================#

#Sparse each word in the file
def split_words(file_name):
	
	#Read the file
	file_ptr = open(file_name, 'r')
	file_data = file_ptr.read()
	file_ptr.close()

	file_data = file_data.lower()

	#Remove Special Character
	for char in spl_char:	
		file_data = file_data.replace(char, ' ')

	#if Prune == yes
	if(prune=='yes'):
		for word in stop_words:
			file_data = file_data.replace((' '+word+' '), ' ')

	#Retrieve list of words from the file data		
	plu_words = file_data.split()

	#Do stemming
	temp_words = [ stemmer.stem(word) for word in plu_words ]

	return temp_words
	
#==========================================================#

#count occurences of each words in a file and match it with the feature then return the occurence list

def get_list(file_type, file_name):

	global features, spam_id, ham_id
		
	file_words = split_words(file_name)
	tmp_unq_word  = set(file_words)
	
	temp_list = [ ]	
	temp_dict = { }

	for word in tmp_unq_word:
		temp_dict[word] = file_words.count(word)	

	#Update the list with the occurence number	
	for word in features:
		if(word in tmp_unq_word):
			temp_list.append(temp_dict[word])		
		else:
			temp_list.append(0)

	#Update W column and Class value
	temp_list.insert(0,1)	# Set x0 as 1 always

	#Update Class label value
	if(file_type == 'spam'):	
		temp_list.insert(-1, spam_id)	
	else:
		temp_list.insert(-1, ham_id)

	
	return temp_list

#==========================================================#

def calc_delta_w(t, o, x):

	global neeta;
	
	delta_w = Decimal(neeta) * ( Decimal(t) - Decimal(o) ) * Decimal(x)

	return delta_w

#==========================================================#

#Calculate Activation values
def calc_activation(file_name):

	temp_list = data_dict[file_name]	

	summation = 0

	for (i,f) in enumerate(features[:-1]):
		summation = summation + (w_random[i] * temp_list[i])

	if summation>0:
		return 1

	else:
		return 0


#==========================================================#

#Update the Weight vector

def update_weight(file_type, file_name):

	predict_O = calc_activation(file_name)	

	if(predict_O==features[-1]):
		pass

	else:
		temp_list = data_dict[file_name]
		for (i,f) in enumerate(features[:-1]):
			delta_w = calc_delta_w(temp_list[-1], predict_O, temp_list[i])
			w_random[i] = Decimal(w_random[i]) + Decimal(delta_w)	
		

#==========================================================#

#Perceptrion Training Rule

def perceptron_training():

	global features_cnt

	loop = 0

	while def_step > loop :

		loop+=1			

		for name in train_spam_file_list: 
			file_name = train_spam_data_path + name
			update_weight('spam', file_name)		


		for name in train_ham_file_list: 
			file_name = train_ham_data_path + name
			update_weight('ham', file_name)

				
	return

#==========================================================#

#Determine the file spam or ham

def determine(file_name):

	temp_list = split_words(file_name)
	un_list = list(set(temp_list))
	
	temp_dict = { }

	for word in un_list:
		temp_dict[word] = temp_list.count(word)

	summation = Decimal(0)
	for word in un_list:
		if(word in features):
			idx = features.index(word)
			summation = summation + (w_random[idx] * temp_dict[word])	

	summation = summation + w_random[0]	
	
	if(summation>0):
		return 1	#Spam
	else:
		return 0	#Ham	


#==========================================================#
#        MAIN PROGRAM
#==========================================================#

#Take folder name from input parameter
input_parm = sy.argv

#Get the training, test data path names from input parameter
train_data_path = input_parm[1]
test_data_path = input_parm[2]
prune = input_parm[3]
#neeta = float(input_parm[4])
#def_step = int(input_parm[5])
#case = int(input_parm[6])

#train_data_path = "/home/selva/Desktop/Spring 2018/CS580L Machine Learning/4. Homework/HW 3/train(copy)"
#test_data_path  = "/home/selva/Desktop/Spring 2018/CS580L Machine Learning/4. Homework/HW 3/test(copy)"
#prune = "no"


#Construct training, test data path names
train_spam_data_path = train_data_path + '/spam/'
train_ham_data_path = train_data_path + '/ham/'

test_spam_data_path = test_data_path + '/spam/'
test_ham_data_path = test_data_path + '/ham/'

#Get the list of file names from training data set
train_spam_file_list = os.listdir(train_spam_data_path)
train_ham_file_list  = os.listdir(train_ham_data_path)
train_file_list = train_spam_file_list + train_ham_file_list

train_spam_file_cnt = len(train_spam_file_list)
train_ham_file_cnt = len(train_ham_file_list)

train_file_cnt = len(train_file_list)

test_spam_file_list = os.listdir(test_spam_data_path)
test_ham_file_list  = os.listdir(test_ham_data_path)

test_spam_file_cnt = len(test_spam_file_list)
test_ham_file_cnt = len(test_ham_file_list)

tot_test_file_cnt = test_spam_file_cnt + test_ham_file_cnt

#Process each file in Training Data Set - Spam Folder
for name in train_spam_file_list: 
	file_name = train_spam_data_path + name
	tot_words = tot_words + (split_words(file_name))	
	total_list.append(file_name)

#Process each file in Training Data Set - Ham Folder
for name in train_ham_file_list: 
	file_name = train_ham_data_path + name
	tot_words = tot_words + (split_words(file_name))
	total_list.append(file_name)


tot_words_cnt = len(tot_words)
temp = list(set(tot_words))
features = w_list + temp + c_list
features_cnt = len(features)

#Build the Matrix - Spam file
for name in train_spam_file_list: 
	file_name = train_spam_data_path + name
	data_dict[file_name] = get_list('spam', file_name)

#Build the Matrix - Ham file
for name in train_ham_file_list: 
	file_name = train_ham_data_path + name
	data_dict[file_name] = get_list('ham', file_name)


#Get W[] with random values between 0 to 1
w_random = [r.randint(-100, 100)/100 for i in range(features_cnt-1)]

#Calculate weight vector
perceptron_training()


#Process the test file data
det_spam_cnt = 0
det_ham_cnt = 0

for name in test_spam_file_list: 
	file_name = test_spam_data_path + name
	check = determine(file_name)
	if(check==1):
		det_spam_cnt +=1 


for name in test_ham_file_list: 
	file_name = test_ham_data_path + name
	check = determine(file_name)
	if(check==0):
		det_ham_cnt +=1


det_tot = det_spam_cnt + det_ham_cnt	

accuracy = (det_tot/tot_test_file_cnt)*100

#Create an output file
file_out = open('Perceptron_stat.txt', 'a')

#file_out.write('===========================\n')
file_out.write("Perceptron Algorithm. \n")

if(prune=="yes"):
	file_out.write("Removed stop words: Yes \n")
else:
	file_out.write("Removed stop words: No \n")
	
file_out.write("Accuracy: ")
file_out.write(str(accuracy))

file_out.write('\n===========================\n')
file_out.close()

print "---------------------------------------"
print "Accuracy of prediction: " + str(accuracy)
print "---------------------------------------"

print(" *** --- End --- *** ")


#==========================================================#









