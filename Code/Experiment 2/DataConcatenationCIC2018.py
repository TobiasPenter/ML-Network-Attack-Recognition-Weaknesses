import pandas as pd

# Read the files
file1 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-14-2018.csv") 
file2 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-15-2018.csv", header=None)
file3 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-16-2018.csv", header=None)
file5 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-21-2018.csv", header=None)
file6 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-22-2018.csv", header=None)
file7 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-23-2018.csv", header=None)
file8 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/02-28-2018.csv", header=None)
file9 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/03-01-2018.csv", header=None)
file10 = pd.read_csv("Data for ML/CIC/CIC-IDS2018/03-02-2018.csv", header=None)

columns = file1.columns

file2.columns = columns
file3.columns = columns
file5.columns = columns
file6.columns = columns
file7.columns = columns
file8.columns = columns
file9.columns = columns
file10.columns = columns
# Concatenate the files
data = pd.concat([file1, file2, file3, file5, file6, file7, file8, file9, file10])

data.columns = columns

# Save the new file
data.to_csv("Data for ML/CIC/CIC-IDS2018.csv")