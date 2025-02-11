import pandas as pandas
import numpy as np
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
total_Dataset = pandas.concat([file1, file2, file3, file4])

total_Dataset.columns = columns

total_Dataset.to_csv('Data for ML/UNSW-NB15/UNSW-NB15_DataSet.csv')

#Remove nan values
total_Dataset["attack_cat"] = total_Dataset["attack_cat"].fillna("Unknown")
total_Dataset["dsport"] = pandas.to_numeric(total_Dataset['dsport'], errors='coerce').fillna(0).astype(int)
total_Dataset["sport"] = pandas.to_numeric(total_Dataset['sport'], errors='coerce').fillna(0).astype(int)

# Not encoding "proto" or "sevices" as they were not used in the model I am replicating
state_Categories = total_Dataset["state"].unique()
attack_Categories = total_Dataset["attack_cat"].unique()

ordinal_Categories = [state_Categories, attack_Categories]

ordinal_Columns = ["state", "attack_cat"]

encoder = make_column_transformer((OrdinalEncoder(categories=ordinal_Categories), ordinal_Columns))

encoded_Data = encoder.fit_transform(total_Dataset)

# Removed "sloss", "trans-depth", "ct-ftp-cmd", "is-ftp-login", "Label" and "ct-flow-http-mthd" as they were unused in the model I am replicating
unchanged_Columns = ["srcip", "sport", "dstip", "dsport", "dur", "sbytes", "dbytes", "sttl", "dttl", "dloss", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"]
unchanged_Data = total_Dataset[unchanged_Columns].values

total_Encoded_Data = np.concatenate((unchanged_Data, encoded_Data), axis=1)
total_Encoded_Columns = unchanged_Columns + ordinal_Columns

total_DataFrame = pandas.DataFrame(total_Encoded_Data, columns=total_Encoded_Columns)

columns_To_Normalise = [col for col in total_DataFrame.columns if col != 'is_sm_ips_ports' and col != 'srcip' and col != 'dstip' and col != 'attack_cat']
total_DataFrame[columns_To_Normalise] = MinMaxScaler().fit_transform(total_DataFrame[columns_To_Normalise])

total_DataFrame.to_csv('Data for ML/UNSW-NB15/UNSW-NB15_DataFrame.csv', index=False)
print("Data Preped")