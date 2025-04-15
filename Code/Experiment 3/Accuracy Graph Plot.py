from matplotlib import pyplot as plt
import numpy as np
import pandas as pandas
from sklearn.model_selection import train_test_split
import pickle

model_For_Testing = pickle.load(open('Models/LightGBM3.pk1', 'rb'))

trainingData = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_DataFrame.csv")
target = trainingData['attack_cat'].values.astype(int)
'''trainingDataColumns = ['attack_cat', 'swin', 'dwin', 'dmeansz', 'dsport', 'Flow Bytes/s', 'proto', 'Spkts', 'Dpkts', 'tcprtt']
trainingData = trainingData[trainingDataColumns]'''

input_train, input_test, target_train, target_test = train_test_split(trainingData.drop(columns='attack_cat'), target, test_size=0.9, random_state=42)

expected_features = model_For_Testing.feature_names_in_

train_features = set(expected_features)
test_features = set(input_test.columns)

missing_features = train_features - test_features
extra_features = test_features - train_features

print("Missing in test:", missing_features)
print("Extra in test:", extra_features)

renameMapping = {
    'Flow Bytes/s': 'Flow0Bytes/s',
    'Flow IAT Mean': 'Flow0IAT0Mean'
}

input_test.rename(columns=renameMapping, inplace=True)

input_test = input_test[expected_features]

attackPrediction = model_For_Testing.predict(input_test)
AllPredictions = np.array(attackPrediction).astype(int)
AllAttacks = np.array(target_test).astype(int)

correct_counts = np.bincount(AllAttacks[AllAttacks == AllPredictions], minlength=10)
incorrect_counts = np.bincount(AllPredictions[AllAttacks != AllPredictions], minlength=10)

total_counts = correct_counts + incorrect_counts
correct_percentages = np.divide(correct_counts, total_counts, where=total_counts != 0) * 100
incorrect_percentages = np.divide(incorrect_counts, total_counts, where=total_counts != 0) * 100

bar_width = 0.5
index = np.arange(10)

plt.bar(index, correct_percentages, bar_width, label='Correct Predictions', color='green')
plt.bar(index, incorrect_percentages, bar_width, bottom=correct_percentages, label='Incorrect Predictions', color='red')

plt.xlabel('Attack Category')
plt.ylabel('Percentage')
plt.title('Correct and Incorrect Prediction Percentages by Attack Type')
plt.xticks(index, ['Unknown', 'Exploits', 'Reconnaissance', 'DoS', 'Generic', 'Shellcode', 'Fuzzers', 'Worms', 'Backdoor', 'Analysis'], rotation=45)
plt.legend()
plt.savefig('Explainer Charts/Summary/On Test/Model 3/Accuracy.png')