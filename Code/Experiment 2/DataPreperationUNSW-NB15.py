import pandas as pandas
import numpy as np
import socket
import struct
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# Load the data
file1 = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_1.csv", header=None) 
file2 = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_2.csv", header=None)
file3 = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_3.csv", header=None)
file4 = pandas.read_csv("Data for ML/UNSW-NB15/UNSW-NB15_4.csv", header=None)

columns = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"]

file1.columns = columns
file2.columns = columns
file3.columns = columns
file4.columns = columns

# Merge data
totalDataset = pandas.concat([file1, file2, file3, file4])

totalDataset.columns = columns

columnsToDrop = ['state', 'srcip', 'sport', 'dstip', 'state', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'stcpb', 'dtcpb', 'trans_depth', 'res_bdy_len', 'tcprtt', 'is_sm_ips_ports', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

totalDataset.drop(columns=columnsToDrop)

totalDataset.to_csv('Data for ML/UNSW-NB15/UNSW-NB15_DataSet.csv')

#Remove nan values
totalDataset["attack_cat"] = totalDataset["attack_cat"].fillna("Unknown")
totalDataset["dsport"] = pandas.to_numeric(totalDataset['dsport'], errors='coerce').fillna(0).astype(int)

# Not encoding "proto" or "sevices" as they were not used in the model I am replicating
attackCategories = totalDataset["attack_cat"].unique()

ordinalCategories = [attackCategories]

ordinalColumns = ["attack_cat"]

encoder = make_column_transformer((OrdinalEncoder(categories=ordinalCategories), ordinalColumns))

encodedData = encoder.fit_transform(totalDataset)

# Removed "sloss", "trans-depth", "ct-ftp-cmd", "is-ftp-login", "Label" and "ct-flow-http-mthd" as they were unused in the model I am replicating
unchangedColumns = ["dsport", "dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "smeansz", "dmeansz", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "synack", "ackdat"]
unchangedData = totalDataset[unchangedColumns].values

totalEncodedData = np.concatenate((unchangedData, encodedData), axis=1)
totalEncodedColumns = unchangedColumns + ordinalColumns

totalDataFrame = pandas.DataFrame(totalEncodedData, columns=totalEncodedColumns)

columnsToNormalise = [col for col in totalDataFrame.columns if col != 'is_sm_ips_ports' and col != 'attack_cat']
totalDataFrame[columnsToNormalise] = MinMaxScaler().fit_transform(totalDataFrame[columnsToNormalise])

print(totalDataFrame.shape)
print(totalDataFrame.columns)

totalDataFrame.to_csv('Data for ML/UNSW-NB15/UNSW-NB15_DataFrame.csv', index=False)
print("Data Preped")