import numpy as np
import csv

# Open training file and extract data
train_file = open("train.csv", "r")
train_csv_reader = csv.reader(train_file)
next(train_csv_reader)

data = []
for row in train_csv_reader:
    data.append(row)

data = np.array(data)
train_file.close()

# Divide the fares into equal size brackets.
highest_fare = 40

# Covert all the fares above highest_fare to a value 39, so that it is in highest fare bracket.
data[data[:,9].astype(np.float) >= highest_fare, 9 ] = highest_fare - 1.0

bracket_size = 10
number_of_brackets = int(highest_fare / bracket_size)

number_of_classes = len(np.unique(data[:,2]))

# Create a 3 dimensional array for gender, number of classes and number of brackets
survival_table = np.zeros((2, number_of_classes, number_of_brackets))

# Find all passengers who fall in each entry of survival table.
for i in range(number_of_classes):
    for j in range(number_of_brackets):
        female_stats = data[(data[:,4] == "female") &
                            (data[:,2].astype(np.float) == i+1) &
                            (data[:,9].astype(np.float) >= j * bracket_size) &
                            (data[:,9].astype(np.float) < (j+1) * bracket_size), 1]

        male_stats = data[(data[:, 4] == "male") &
                          (data[:, 2].astype(np.float) == i + 1) &
                          (data[:, 9].astype(np.float) >= j * bracket_size) &
                          (data[:, 9].astype(np.float) < (j + 1) * bracket_size), 1]

        survival_table[0,i,j] = np.mean(female_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(male_stats.astype(np.float))

survival_table[survival_table != survival_table] = 0
print(survival_table)

survival_table[survival_table >= 0.5] = 1
survival_table[survival_table < 0.5] = 0
print(survival_table)

# Open test file and predict the values.
test_file = open("test.csv", "r")
test_csv_reader_obj = csv.reader(test_file)
next(test_csv_reader_obj)

# Open prediction file.
gen_class_model_pred_file = csv.writer(open("genderclassmodel_1.csv", "w", newline=''))

# Write header into the file.
gen_class_model_pred_file.writerow(["PassengerID", "Survived"])

for row in test_csv_reader_obj:
    for j in range(number_of_brackets):

        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break

        if row[8] > highest_fare:
            bin_fare = number_of_brackets - 1
            break
        if (row[8] >= j * bracket_size) and (row[8] < (j + 1) * bracket_size):
            bin_fare = j
            break

    if row[3] == 'female':  # If the passenger is female
        gen_class_model_pred_file.writerow([row[0], "%d" % int(survival_table[0, float(row[1]) - 1, bin_fare])])
    else:  # else if male
        gen_class_model_pred_file.writerow([row[0], "%d" % int(survival_table[1, float(row[1]) - 1, bin_fare])])

test_file.close()
