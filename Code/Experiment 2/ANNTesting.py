import random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pandas
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
import pickle
import matplotlib.backends.backend_agg as agg


# Load Model
model_For_Testing = pickle.load(open('Models/MLPModel.pk1', 'rb'))

# Load data
trainingData = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_DataFrame.csv")

# Testing with Random Data|
print ("Testing with Random data")
# Gerate Random Data
numRows = 200000
columnNames = trainingData.columns
randomData = np.random.rand(numRows, len(columnNames) - 1)
attackCatData = np.random.randint(0, 14, size=(numRows, 1))
randomDataFull = np.concatenate((randomData, attackCatData), axis=1)
randomDF = pd.DataFrame(randomDataFull, columns=columnNames)

target = randomDF['attack_cat'].values.astype(float)

# Data split
input_training, input_test, target_training, target_test = train_test_split(randomDF.drop(columns='attack_cat'), target, test_size=0.2, random_state=42)

AllPredictions= []
AllAttacks = []

# Test the model and give statistics
for i in range(5):
    attackPrediction = model_For_Testing.predict(input_test)
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
    
    # Resplit data
    ran_state_val = random.randint(30, 70)
    input_training, input_test, target_training, target_test = train_test_split(randomDF.drop(columns='attack_cat'), target, test_size=0.2, random_state=ran_state_val)
    
# Overall statisticss 
AllAttacks = np.concatenate(AllAttacks, axis=0)
AllPredictions = np.concatenate(AllPredictions, axis=0)

accuracy = accuracy_score(AllAttacks, AllPredictions)
precision = precision_score(AllAttacks, AllPredictions, average="micro")
recall = recall_score(AllAttacks, AllPredictions, average="micro")
f1 = f1_score(AllAttacks, AllPredictions, average="micro")
errorMean = mean_squared_error(AllAttacks, AllPredictions)
errorAbsolute = mean_absolute_error(AllAttacks, AllPredictions)
r2 = r2_score(AllAttacks, AllPredictions)

print("Total Accuracy Score: ",accuracy,"\nTotal Precision Score: ", precision,"\nTotal Recall Score:", recall,"\nTotal F1 Score: ", f1,"\nTotal Mean Square Error: ", errorMean,"\nTotal Mean Absolute Error: ", errorAbsolute,"\nTotal R2 Score: ", r2,"\n")

# Testing with dataset
print("Testing with Dataset")

trainingData = pandas.DataFrame(trainingData.values, columns=trainingData.columns)

# Define the inputs and targets
trainingData = trainingData.sample(n=200000, random_state=42)
target = trainingData['attack_cat'].values.astype(float)

# Data split
input_training, input_test, target_training, target_test = train_test_split(trainingData.drop(columns='attack_cat'), target, test_size=0.2, random_state=42)

AllPredictions= []
AllAttacks = []

# Test the model and give statistics
for i in range(5):
    attackPrediction = model_For_Testing.predict(input_test)
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
    
    # Resplit data
    ran_state_val = random.randint(30, 70)
    input_training, input_test, target_training, target_test = train_test_split(trainingData.drop(columns='attack_cat'), target, test_size=0.2, random_state=ran_state_val)
   
# Overall statisticss 
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

# Initialise shap plot
shap.initjs()

# Make explainer
background_sample = shap.sample(input_test, 10000)

explainer = shap.Explainer(model_For_Testing.predict_proba, background_sample)

shap_values = explainer.shap_values(background_sample)

shap_values = np.transpose(shap_values.transpose(1, 0, 2))

# Shape values for each class
for i, class_shap_values in enumerate(shap_values):
    print(f"Generating summary plot for class {i}")
    summary_plot = shap.summary_plot(class_shap_values, background_sample, feature_names=background_sample.columns, show=False)
    
    fig = plt.gcf()

    # Use agg backend to save it as PNG
    png_file_path = f"Explainer Charts/Summary/On Build/Model 1/shap_plot_class{i}.png"
    agg_backend = agg.FigureCanvasAgg(fig)
    agg_backend.print_png(png_file_path)
    plt.close(fig)