import random
import numpy as np
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
import pickle
from imblearn.over_sampling import SMOTE

# Load data
trainingData = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_DataFrame.csv")

# Define the inputs and targets
target = trainingData['attack_cat'].values.astype(float)

# Data split
input_training, input_test, target_training, target_test = train_test_split(trainingData.drop(columns='attack_cat'), target, test_size=0.1, random_state=42)

# Use SMOTE to balance
smote = SMOTE(sampling_strategy='auto', random_state=42)
input_training, target_training = smote.fit_resample(input_training, target_training)

# Not using an Random Forest for project but it is being used as a test
MLPAttackClassifier = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 16), activation='relu', solver='adam', max_iter=500, random_state=42, early_stopping=True)
MLPAttackClassifier.fit(input_training, target_training)

AllPredictions= []
AllAttacks = []

#Test model and generate statistics
for i in range(5):
    attackPrediction = MLPAttackClassifier.predict(input_test)
    AllPredictions.append(attackPrediction)
    AllAttacks.append(target_test)
    
    accuracy = accuracy_score(target_test, attackPrediction)
    precision = precision_score(target_test, attackPrediction, average="micro")
    recall = recall_score(target_test, attackPrediction, average="micro")
    f1 = f1_score(target_test, attackPrediction, average="micro")
    errorMean = mean_squared_error(target_test, attackPrediction)
    errorAbsolute = mean_absolute_error(target_test, attackPrediction)
    r2 = r2_score(target_test, attackPrediction)
    
    print("Accuracy Score: ",accuracy,"\nPrecision Score: ", precision,"\nRecall Score:", recall,"\nF1 Score: ", f1,"\nMean Square Error: ", errorMean,"\nMean Absolute Error: ", errorAbsolute,"\nR2 Score: ",r2, "\n\n\n")    
    
    old_test_input = input_test
    old_test_target = target_test
    ran_state_val = random.randint(30, 70)
    input_training, input_test, target_training, target_test = train_test_split(trainingData.drop(columns='attack_cat'), target, test_size=0.1, random_state=ran_state_val)
    
#Overall statistics
AllAttacks = np.concatenate(AllAttacks, axis=0)
AllPredictions = np.concatenate(AllPredictions, axis=0)

accuracy = accuracy_score(AllAttacks, AllPredictions)
precision = precision_score(AllAttacks, AllPredictions, average="micro")
recall = recall_score(AllAttacks, AllPredictions, average="micro")
f1 = f1_score(AllAttacks, AllPredictions, average="micro")
errorMean = mean_squared_error(AllAttacks, AllPredictions)
errorAbsolute = mean_absolute_error(AllAttacks, AllPredictions)
r2 = r2_score(AllAttacks, AllPredictions)

print("Total Accuracy Score: ",accuracy,"\nTotal Precision Score: ", precision,"\nTotal Recall Score:", recall,"\nTotal F1 Score: ", f1,"\nTotal Mean Square Error: ", errorMean,"\nTotal Mean Absolute Error: ", errorAbsolute,"\nTotal R2 Score: ",r2)

#Save model
pickle.dump(MLPAttackClassifier, open("Models/MLPModel.pk1", 'wb'))