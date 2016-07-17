#!/usr/bin/env python
from bs4 import BeautifulSoup        
import numpy as np     
import re
import nltk
from sklearn import cross_validation 
#nltk.download()
import operator
from nltk.corpus import stopwords
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

raw_data = [ records for records in open('classification_train.tsv','r')]

print "total data size is {}".format(len(raw_data))


def generate_data_labels(raw_data):
	data=[]
	labels=[]
	unique_labels = set()
	for records in raw_data:
		raw = records.strip().split("\t")
		data.append(raw[0]+" "+raw[2])
		labels.append(raw[1])
		unique_labels.add(raw[1])
	label_dictionary = dict((v,k) for k,v in enumerate(unique_labels))
	return data,labels,label_dictionary

def get_indices_from_labels(labels,label_dictionary):
	label_indices=[]	
	for label in labels:
		label_indices.append(label_dictionary.get(label,"N/A"))
	return label_indices
		
"""
def divide_train_val(data,ratio=0.8):
	overall_label_dictionary=generate_label_dictionary(data)
	random.shuffle(data)
	total=len(data)
	train_length=int(ratio*total)
	train_data=data[:train_length]
	val_data=data[train_length:]
	return train_data,val_data,overall_label_dictionary

"""
sample_size = 4000

reduced_data = [ raw_data[i] for i in sorted(random.sample(xrange(len(raw_data)), sample_size)) ]

data,raw_labels,label_dictionary=generate_data_labels(reduced_data)
label_indices = get_indices_from_labels(raw_labels,label_dictionary)

print "Data ..."
print data
print "labels ..."
print raw_labels
print "label indices ..."
print label_indices
print "label dictionary ..."
print label_dictionary


# sort dictionary in ascending order by value and get 10 minimum from top 10
t = sorted(label_dictionary.items(), key=lambda x:x[1])[:sample_size]

for x in t:
     print "{0}: {1}".format(*x)

def description_to_words( raw_review ):
	# Function to convert a raw review to a string of words
	# The input is a single string (a raw product description), and 
	# the output is a single string (a preprocessed product description)
	#
	# 1. Remove HTML
	review_text = BeautifulSoup(raw_review,"lxml").get_text() 
	#
	# 2. Remove non-letters        
	letters_only = re.sub("[^0-9a-zA-Z]", " ", review_text) 
	#
	# 3. Convert to lower case, split into individual words
	words = letters_only.lower().split()                             
	#
	# 4. In Python, searching a set is much faster than searching
	#   a list, so convert the stop words to a set
	stops = set(stopwords.words("english"))                  
	# 
	# 5. Remove stop words
	meaningful_words = [w for w in words if not w in stops]   
	#6. Remove stemming
	porter = nltk.PorterStemmer()
	meaningful_words = [porter.stem(t) for t in meaningful_words]
	#
	# 7. Join the words back into one string separated by space, 
	# and return the result.
	return( " ".join( meaningful_words )) 


print "Cleaning and parsing the training set product descriptions...\n"
# Initialize an empty list to hold the clean reviews
clean_product_description = []

# Loop over each review; create an index i that goes from 0 to the length
# of the product description list 
for i in xrange( 0, sample_size ):
	# Call our function for each one, and add the result to the list of
	if( (i+1)%100 == 0 ):
		print "cleaned record %d of %d\n" % ( i+1, sample_size)  
	# clean reviews
	clean_product_description.append( description_to_words( data[i]) )


print "Creating the bag of words...\n"

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
"""
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 10000) 
"""
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
#train_data_features = vectorizer.fit_transform(clean_product_description)

vectorizer = TfidfVectorizer(min_df=1)
train_data_features = vectorizer.fit_transform(clean_product_description)

print train_data_features.shape




#exit()

# Take a look at the words in the vocabulary
#vocab = vectorizer.get_feature_names()
#print vocab

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
#for tag, count in zip(vocab, dist):
#    print count, tag

from sklearn.metrics import confusion_matrix

def trainRFC(train, labels):
	print 'starting RFC'
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.2, random_state=1)
	clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
	print 'Training..'
	model = clf.fit(X_train, y_train)
	acc_pred = model.predict(X_test)
	accuracy = model.score(X_test, y_test, sample_weight=None)
	print 'Accuracy of RFC: ', accuracy*100
	return model

def testRFC(test_data,model):
    print 'Testing..'
    rfc_predictions = model.predict(test)
    rfc_probs = model.predict_proba(test)
    rfc_bestProbs = rfc_probs.max(axis=1)
    print 'done with RFC'
    return rfc_predictions, rfc_bestProbs


def trainSVC(train, labels):
	print 'starting SVC'
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, labels, test_size=0.3, random_state=0)
	clf = SVC(C=1000000.0, gamma=0.0, kernel='rbf',probability=True)
	print 'Training..'
	model = clf.fit(X_train, y_train)
	acc_pred = model.predict(X_test)
	accuracy = model.score(X_test, y_test, sample_weight=None)
	print("Accuracy of SVC: %f ", accuracy*100)
	print(confusion_matrix(acc_pred, y_test))

def testSVC(test_data,model):
    print 'Testing..'
    svc_predictions = clf.predict(test)
    svc_probs = clf.predict_proba(test)
    svc_bestProbs = svc_probs.max(axis=1)
    print 'done with SVC'
    return svc_predictions, svc_bestProbs

trainSVC(train_data_features,label_indices)

