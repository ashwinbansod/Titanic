import pandas as pd
import numpy as np
import csv
import pylab as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


def read_file(filename):
    input_df = pd.read_csv(filename, header=0)
    return input_df


def map_gender(input_df):
    male = 0
    female = 1
    input_df['Gender'] = input_df['Sex'].map({'female' : female, 'male' : male}).astype(int)


def map_embarkpoint(input_df):
    input_df['EmbarkPoint'] = input_df['Embarked'].map({'C' : 1, 'S' : 2, 'Q' : 3})
    input_df.loc[(input_df['EmbarkPoint'].isnull())] = 2


def calculate_median_ages(input_df, median_ages):
    for gender in range(0,2):
        for pclass in range(0,3):
            median_ages[gender, pclass] = input_df['Age'].dropna()[(input_df['Gender'] == gender) &
                                                                   (input_df['Pclass'] == pclass+1)].median()


def update_age(input_df, median_ages):
    input_df['AgeFill'] = input_df['Age']

    for gender in range(0,2):
        for pclass in range(0,3):
            input_df.loc[(input_df['Age'].isnull()) &
                         (input_df['Gender'] == gender) &
                         (input_df['Pclass'] == pclass+1), 'AgeFill'] = median_ages[gender, pclass]


def calculate_median_fare(input_df, median_fare):
    for gender in range(0, 2):
        for pclass in range(0, 3):
            median_fare[gender, pclass] = input_df['Fare'].dropna()[(input_df['Gender'] == gender) &
                                                                    (input_df['Pclass'] == pclass + 1) &
                                                                    (input_df['Fare'] != 0)].median()


def update_fare(input_df, median_fare):
    input_df['FareFill'] = input_df['Fare']

    for gender in range(0,2):
        for pclass in range(0,3):
            input_df.loc[((input_df['Fare'].isnull()) | (input_df['Fare'] == 0)) &
                         (input_df['Gender'] == gender) &
                         (input_df['Pclass'] == pclass+1), 'FareFill'] = median_fare[gender, pclass]


def create_age_is_null(input_df):
    input_df['AgeIsNull'] = pd.isnull(input_df.Age).astype(int)


def create_family_size(input_df):
    input_df['FamilySize'] = input_df['SibSp'] + input_df['Parch']


def create_extra_fields(input_df):
    # Create Column for Age is null
    create_age_is_null(input_df)

    # Create column for family size, which is sum of number of siblings and number of parents
    create_family_size(input_df)


def prepare_data_for_prediction(input_df, train):
    passngr_id_list = input_df['PassengerId'].values

    # Drop columns which are nt required and which has string data in it.
    temp_df = input_df.drop(['PassengerId', 'Name', 'Sex', 'Fare', 'Ticket', 'Cabin', 'Age', 'Embarked'], axis=1)
    temp_df = temp_df.dropna()

    header_list = []
    if train == True:
        header_list = temp_df.columns.values
        header_list = header_list[1:]

    temp_arr = temp_df.values

    if train == True:
        data_part = temp_arr[:,1:]
        label_or_passngr_list = temp_arr[:,0]
    else:
        data_part = temp_arr
        label_or_passngr_list = passngr_id_list

    return data_part, label_or_passngr_list, header_list


def prepare_data(in_file, train, median_ages, median_fare):
    input_df = read_file(in_file)

    # Add new columns in dataframe with numerical values
    map_gender(input_df)
    map_embarkpoint(input_df)

    # Calculate median of ages and fare based on gender and class
    if train == True:
        calculate_median_ages(input_df, median_ages)
        calculate_median_fare(input_df, median_fare)

    # update the null values of ages
    update_age(input_df, median_ages)

    # Create additional fields
    create_extra_fields(input_df)

    # Update fare values
    update_fare(input_df, median_fare)

    # Prepare final data for prediction
    data_part, label_or_passngr_list, header_list = prepare_data_for_prediction(input_df, train)

    return data_part, label_or_passngr_list, header_list


def write_to_output_file(filename, passngr_id_list, output_data):
    # Open output file
    out_fp = open(filename, "w", newline='')
    out_csv = csv.writer(out_fp)

    # Write Header
    out_csv.writerow(["PassengerID", "Survived"])

    # Write each record in file
    for i in range(len(passngr_id_list)):
        out_csv.writerow([passngr_id_list[i], int(output_data[i])])

    out_fp.close()


def get_feature_importance(rnd_forest, train_data, header_list):
    importances = rnd_forest.feature_importances_

    std = np.std([tree.feature_importances_ for tree in rnd_forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature Importance:")
    for f in range(train_data.shape[1]):
        print("%d. feature [%s] (%f)" % (f + 1, header_list[indices[f]], importances[indices[f]]))


def main():
    train_file = 'train.csv'
    test_file = 'test.csv'
    output_filename = 'rnd_frst_prediction_full_2.csv'

    median_ages = np.zeros((2, 3))
    median_fare = np.zeros((2, 3))
    train_labels = []
    passngr_id_list = []

    # Read Training file and get data
    train_data, train_labels, header_list = prepare_data(train_file, True, median_ages, median_fare)

    # Read Test file and get data
    test_data, passngr_id_list, ignore_this_list = prepare_data(test_file, False, median_ages, median_fare)

    # Create Random forest
    rnd_forest = RandomForestClassifier(n_estimators=100,)

    # Train the classifier
    rnd_forest = rnd_forest.fit(train_data, train_labels)

    # Predict Output
    test_output = rnd_forest.predict(test_data)

    # Write data to output file.
    write_to_output_file(output_filename, passngr_id_list, test_output)

    # Get the importance of features.
    get_feature_importance(rnd_forest, train_data, header_list)


if __name__ == '__main__':
    main()


