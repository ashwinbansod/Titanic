import numpy as np
import csv


train_csv_fp = csv.reader(open("train.csv", "r"))

data = []
for row in train_csv_fp:
    data.append(row)

data = np.array(data)

number_of_passengers = np.size(data[1:,1].astype(np.float))
number_of_survivors = np.sum(data[1:,1] == "1")
portion_survived = float(number_of_survivors/number_of_passengers)
print("Total number of Passangers: " + str(number_of_passengers))
print("Total number of Survivors: " + str(number_of_survivors))
print("Portion of Passangers survived: " + str(portion_survived))



female_stats = (data[:,4] == "female")
male_stats = (data[:,4] == "male")

total_females = len(data[female_stats])
total_males = len(data[male_stats])
female_survived = np.sum(data[female_stats,1].astype(np.float))
male_survived = np.sum(data[male_stats,1].astype(np.float))
portion_of_female_survived = float(female_survived / total_females)
portion_of_males_survived = float(male_survived/ total_males)
# print(total_females)
# print(total_males)
# print(female_survived)
# print(male_survived)
print("Proportion of females survived: " + str(portion_of_female_survived))
print("Proportion of males survived: " + str(portion_of_males_survived))


# From these stats you can infer that, if the passenger is a female,
# there are very high chances that passenger survived.
# On the other hand, if the passenger is male, it won't be wrong
# to assume that passenger did not survive.

# Write to gender class model.

# Open Test file.
test_file = open("test.csv", "r")
test_csv_reader = csv.reader(test_file)
next(test_csv_reader)

# Open prediction file.
gen_model_pred_file = csv.writer(open("gendermodel_1.csv", "w", newline=''))

# Write header into the file.
gen_model_pred_file.writerow(["PassengerID", "Survived"])

# for every record in the test file, predict the survival of the passenger.
for rec in test_csv_reader:
    test_rec = np.array(rec)

    if test_rec[3] == "female":
        survived = "1"
    else:
        survived = "0"

    gen_model_pred_file.writerow([test_rec[0], survived])

test_file.close()

