
# coding: utf-8

# In[ ]:

# Dataset Link => https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
# Dataset => https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data

# Step 1 => Get the dataset and load it into a pandas dataframe
import pandas as pd
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", names = ['buying','maint','doors','persons','lug_boot','safety','acceptability'])
# Take a look at what the data looks like
print data


# In[ ]:

# Step 2 => Create a feature set from the data and encode strings
features = pd.get_dummies(data[['buying', 'maint', 'lug_boot', 'safety', 'doors', 'persons']])
labels = pd.get_dummies(data['acceptability'])
# What do features and labels look like?
print 'Features =>\n', features
print 'Labels =>\n', labels


# In[ ]:

# Step 3 => Separate into testing and training sets
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3)

# Verify that the dataset has been split into 2 sets in the appropriate proportion
print len(features_train)
print len(features_test)


# In[ ]:

# Step 4 => Let's classify using DecisionTrees
from sklearn.tree import DecisionTreeClassifier
# Create the classifier
dtc = DecisionTreeClassifier()
# Generate a model by fitting to the training data
dtc.fit(features_train, labels_train)
# Predict the acceptability on the test data
predictions = dtc.predict(features_test)
# Get the statistics(precision, recall, F1, accuracy) for the statistics
print 'Accuracy = ', dtc.score(features_test, labels_test)
from sklearn.metrics import classification_report
print classification_report(labels_test, predictions, target_names = ['Acceptable', 'Good', 'Unacceptable', 'Very Good'])

