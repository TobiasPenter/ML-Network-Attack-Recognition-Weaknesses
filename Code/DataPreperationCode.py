import pandas as pandas
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# Load the data set
dataUncleaned = pandas.read_csv("Data for ML/NF-UNSW-NB15-v2.csv")

# Encode Data
ordinalColumn = ["Attack"]
ordinalColumnCategories = [['Benign', 'Exploits', 'Generic', 'Fuzzers', 'Backdoor', 'DoS', 'Reconnaissance', 'Shellcode', 'Worms', 'Analysis']]
dataUncleaned['Attack'] = dataUncleaned['Attack'].astype(str)
encoding = OrdinalEncoder(categories = ordinalColumnCategories, dtype = int)

# Encode only the 'Attack' column
dataUncleaned['Attack'] = encoding.fit_transform(dataUncleaned[['Attack']])


# Get Unencoded Data
# Removing IP data as it will be relied upon too much
trainingData = dataUncleaned.values


# Create Collumn Names
trainingDataColumns = dataUncleaned.columns

# Created a Data Frame
trainingDataFrame = pandas.DataFrame(trainingData, columns=trainingDataColumns)

# Normalise Collumns
columnsToNormalise = [col for col in trainingDataFrame.columns if col != 'IPV4_SRC_ADDR' and col != 'IPV4_DST_ADDR' and col != 'Attack' and col != 'L4_SRC_PORT' and col != 'L4_DST_PORT' and col != 'L7_PROTO' and col != 'FTP_COMMAND_RET_CODE' and col != 'DNS_QUERY_TYPE']
scaler = MinMaxScaler()
trainingDataFrame[columnsToNormalise] = scaler.fit_transform(trainingDataFrame[columnsToNormalise])

print(dataUncleaned['Attack'].unique())

trainingDataFrame.to_csv('Data for ML/trainingData.csv')
print("Data Preped")