import pandas as pd
import numpy as np
import csv
import pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

# Read training data
train_data = pd.read_csv('train.csv', header=0)

# Add new columns in dataframe with numerical values
train_data['Gender'] = train_data['Sex'].map({'female' : 0 , 'male' : 1}).astype(int)
train_data['EmbarkPoint'] = train_data['Embarked'].map({'C' : 1, 'S' : 2, 'Q' : 3})

median_ages = np.zeros((2,3))

# Get median age per Gender and class type.
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = train_data['Age'].dropna()[(train_data['Gender'] == i) &
                                                     (train_data['Pclass'] == j+1)].median()

# Create a Table for calculated Ages.
# Copy original values of ages to the column
train_data['Agefill'] = train_data['Age']

# Fill in appropriate median ages to new column
for i in range(0,2):
    for j in range(0,3):
        train_data.loc[(train_data['Age'].isnull()) & (train_data['Gender'] == i) &
                       (train_data['Pclass'] == j+1), 'Agefill' ] = median_ages[i,j]

# Create some extra columns
train_data['AgeIsNull'] = pd.isnull(train_data.Age).astype(int)
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

# Update incorrect fare values
median_fare = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_fare[i,j] = train_data['Fare'].dropna()[(train_data['Gender'] == i) &
                                                       (train_data['Pclass'] == j+1) &
                                                       (train_data['Fare'] != 0)].median()
print(median_fare)

train_data['FareFill'] = train_data['Fare']
for i in range(0,2):
    for j in range(0,3):
        train_data.loc[((train_data['Fare'].isnull()) | (train_data['Fare'] == 0)) &
                       (train_data['Gender'] == i) &
                       (train_data['Pclass'] == j+1), 'FareFill'] = median_fare[i, j]

# print(train_data[(train_data['Fare'] == 0)]['FareFill'])

# Drop columns which are nt required and which has string data in it.
train_data = train_data.drop(['PassengerId', 'Name', 'Sex', 'Fare', 'Ticket', 'Cabin', 'Age', 'Embarked'], axis=1)
train_data = train_data.dropna()

# Get column headers for plotting graph
train_header = train_data.columns.values
train_header = train_header[1:]

# Convert dataframe into numpy array.
train_data = train_data.values

# Split training data into train and test data.
train_part, test_part, train_label, test_label = train_test_split(train_data[:,1:], train_data[:,0],
                                                                  test_size=0.2, random_state=50)

# Read actual test data and prepare for prediction.
orig_test_data = pd.read_csv("test.csv")
orig_test_data['Gender'] = orig_test_data['Sex'].map({'female' : 0 , 'male' : 1}).astype(int)
orig_test_data['EmbarkPoint'] = orig_test_data['Embarked'].map({'C' : 1, 'S' : 2, 'Q' : 3})

# Create a Table for calculated Ages.
# Copy original values of ages to the column
orig_test_data['Agefill'] = orig_test_data['Age']

# Fill in appropriate median ages to new column
for i in range(0,2):
    for j in range(0,3):
        orig_test_data.loc[(orig_test_data['Age'].isnull()) & (orig_test_data['Gender'] == i) &
                           (orig_test_data['Pclass'] == j + 1), 'Agefill'] = median_ages[i, j]

# Create some extra columns
orig_test_data['AgeIsNull'] = pd.isnull(orig_test_data.Age).astype(int)
orig_test_data['FamilySize'] = orig_test_data['SibSp'] + orig_test_data['Parch']

# Substitute the null Fare values by 0
# orig_test_data.loc[(orig_test_data['Fare'].isnull()), 'Fare'] = 0

# Update incorrect fare values
orig_test_data['FareFill'] = orig_test_data['Fare']
for i in range(0,2):
    for j in range(0,3):
        orig_test_data.loc[((orig_test_data['Fare'].isnull()) | (orig_test_data['Fare'] == 0)) &
                       (orig_test_data['Gender'] == i) &
                       (orig_test_data['Pclass'] == j+1), 'FareFill'] = median_fare[i, j]


# Drop columns which are nt required and which has string data in it.
test_data = orig_test_data.drop(['PassengerId', 'Name', 'Sex', 'Fare', 'Ticket', 'Cabin', 'Age', 'Embarked'], axis=1)
test_data = test_data.dropna()

# Convert dataframe into numpy array.
test_data = test_data.values

# Create Random forest
rnd_forest = RandomForestClassifier(n_estimators=100,)

# Train the classifier
rnd_forest = rnd_forest.fit(train_part, train_label)

# Predict Output
output = rnd_forest.predict(test_part)

accuracy = accuracy_score(test_label, output)
print("Train Test accuracy: " + str(accuracy))

test_output = rnd_forest.predict(test_data)

# Open output submission file
out_file = open("rnd_frst_prediction_2.csv", "w", newline='')
out_csv = csv.writer(out_file)


orig_test_data = orig_test_data.values
out_csv.writerow(["PassengerID", "Survived"])
for i in range(len(test_output)):
    out_csv.writerow([orig_test_data[i][0], int(test_output[i])])

importances = rnd_forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in rnd_forest.estimators_],
             axis=0)
print(std)
indices = np.argsort(importances)[::-1]

for f in range(train_part.shape[1]):
    print("%d. feature [%s] (%f)" % (f + 1, train_header[indices[f]], importances[indices[f]]))

indices_header = [train_header[x] for x in indices]

# # Plot the feature importance of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(train_part.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
#
# plt.xticks(range(train_part.shape[1]), indices_header)
# plt.xlim([-1, train_part.shape[1]])
# plt.show()