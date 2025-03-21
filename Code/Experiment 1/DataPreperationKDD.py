import pandas as pandas
import numpy as np
import socket
import struct
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# Load the data
file1 = pandas.read_csv("Data for ML/NSL-KDD/KDDTrain.csv", header=None) 
file2 = pandas.read_csv("Data for ML/NSL-KDD/KDDTest.csv", header=None)

columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class", "difficulty"]

file1.columns = columns
file2.columns = columns

# Merge data
total_Dataset = pandas.concat([file1, file2])

total_Dataset.columns = columns

total_Dataset["class"] = total_Dataset["class"].replace(["apache2", "mailbomb", "processtable", "udpstorm", "back", "land", "neptune", "pod", "teardrop", "smurf"], "dos")
total_Dataset["class"] = total_Dataset["class"].replace(["httptunnel", "xterm", "ps", "sqlattack", "saint", "mscan", "buffer_overflow", "loadmodule", "perl", "rootkit"], "u2r")
total_Dataset["class"] = total_Dataset["class"].replace(["named", "sendmail", "snmpgetattack", "snmpguess", "worm", "xsnoop", "xlock", "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster"], "r2l")
total_Dataset["class"] = total_Dataset["class"].replace(["ipsweep", "nmap", "portsweep", "satan"], "probe")

total_Dataset.to_csv('Data for ML/NSL-KDD/KDD.csv')

# NEncoding ordinal collumns
protocolType_Categories = total_Dataset["protocol_type"].unique()
service_Categories = total_Dataset["service"].unique()
flag_Categories = total_Dataset["flag"].unique()
class_Categories = total_Dataset["class"].unique()

print(total_Dataset["class"].unique())

ordinal_Categories = [protocolType_Categories, service_Categories, flag_Categories, class_Categories]

ordinal_Columns = ["protocol_type", "service", "flag", "class"]

encoder = make_column_transformer((OrdinalEncoder(categories=ordinal_Categories), ordinal_Columns))

encoded_Data = encoder.fit_transform(total_Dataset)

# Removed "difficulty" as it is unnecessary
unchanged_Columns = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
unchanged_Data = total_Dataset[unchanged_Columns].values

total_Encoded_Data = np.concatenate((unchanged_Data, encoded_Data), axis=1)
total_Encoded_Columns = unchanged_Columns + ordinal_Columns

total_DataFrame = pandas.DataFrame(total_Encoded_Data, columns=total_Encoded_Columns)

columns_To_Normalise = [col for col in total_DataFrame.columns if col != 'class']
total_DataFrame[columns_To_Normalise] = MinMaxScaler().fit_transform(total_DataFrame[columns_To_Normalise])

total_DataFrame.to_csv('Data for ML/NSL-KDD/KDD_DataFrame.csv', index=False)
print("Data Preped")