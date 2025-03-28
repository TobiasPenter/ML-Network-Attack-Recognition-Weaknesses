import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file2017 = pd.read_csv("Data for ML/CIC/CIC-IDS2017.csv")
file2018 = pd.read_csv("Data for ML/CIC/CIC-IDS2018.csv")

file2017.columns = file2017.columns.str.strip()

columnsToDrop = ['Flow ID', 'Source Port', 'Source IP', 'Destination IP', 'Fwd Header Length.1']
file2017 = file2017.drop(columns=columnsToDrop)

renameMapping = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Fwd Header Len': 'Fwd Header Length',
    'Bwd Header Len': 'Bwd Header Length',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Pkt Len Min': 'Min Packet Length',
    'Pkt Len Max': 'Max Packet Length',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std',
    'Pkt Len Var': 'Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count',
    'Pkt Size Avg': 'Average Packet Size',
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min': 'min_seg_size_forward',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes',
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Bwd Pkts': 'Subflow Bwd Packets'
    
}

file2018.rename(columns=renameMapping, inplace=True)
file2018.columns = file2018.columns.str.strip()

fileCIC = pd.concat([file2017, file2018])

fileCIC.columns = file2018.columns

columnsToDrop = ['Unnamed: 0', 'Timestamp', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 
    'Bwd Packet Length Std', 'Flow Packets/s', 'Flow IAT Std', 'Flow IAT Max', 
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 
    'Bwd IAT Total', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 
    'Max Packet Length', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 
    'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Std', 'Active Max', 'Active Min', 
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Packet Length Mean']

fileCIC = fileCIC.drop(columns=columnsToDrop)

renameMapping = {
    'Destination Port': 'dsport',
    'Protocol': 'proto',
    'Flow Duration': 'dur',
    'Total Fwd Packets': 'Spkts',
    'Total Backward Packets': 'Dpkts',
    'Total Length of Fwd Packets': 'sbytes',
    'Total Length of Bwd Packets': 'dbytes',
    'Fwd Packet Length Mean': 'smeansz',
    'Bwd Packet Length Mean': 'dmeansz',
    'Flow Bytes/s': 'Flow Bytes/s',
    'Flow IAT Mean': 'Flow IAT Mean',
    'Fwd IAT Mean': 'Sintpkt',
    'Bwd IAT Mean': 'Dintpkt',
    'Init_Win_bytes_forward': 'swin',
    'Init_Win_bytes_backward': 'dwin',
    'Active Mean': 'tcprtt',
    'Label': 'attack_cat'
}

fileCIC.rename(columns=renameMapping, inplace=True)

columnOrder = ['dsport', 'dur', 'sbytes', 'dbytes', 'Spkts', 'Dpkts', 'swin', 'dwin', 'smeansz', 'dmeansz', 'Sintpkt', 'Dintpkt', 'Flow IAT Mean', 'Flow Bytes/s', 'tcprtt', 'attack_cat', 'proto']
fileCIC = fileCIC[columnOrder]

fileCIC.to_csv('Data for ML/CIC/CICDatasetCombined.csv')

fileCIC['attack_cat'] = fileCIC['attack_cat'].str.strip()

categoryMapping = {
    "BENIGN": "Unknown",
    "Benign": "Unknown",
    "DDoS": "DoS",
    "PortScan": "Reconnaissance",
    "Bot": "Worms",
    "Infiltration": "Backdoor",
    "Infilteration": "Backdoor",
    "Web Attack \x96 Brute Force": "Exploits",
    "Brute Force -Web": "Exploits",
    "Web Attack \x96 XSS": "Exploits",
    "Brute Force -XSS": "Exploits",
    "Web Attack \x96 Sql Injection": "Exploits",
    "SQL Injection": "Exploits",
    "FTP-Patator": "Fuzzers",
    "FTP-BruteForce": "Fuzzers",
    "SSH-Patator": "Fuzzers",
    "SSH-Bruteforce": "Fuzzers",
    "DoS slowloris": "DoS",
    "DoS attacks-Slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS",
    "DoS Hulk": "DoS",
    "DoS attacks-Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS attacks-GoldenEye": "DoS",
    "Heartbleed": "Shellcode",
    "DDOS attack-LOIC-UDP": "DoS",
    "DDOS attack-HOIC": "DoS",
    "Brute Force": "Fuzzers"
}

fileCIC['attack_cat'] = fileCIC['attack_cat'].replace(categoryMapping)

attackEncoder = {
    'Unknown': 0,
    'Fuzzers': 6,
    'DoS': 3,
    'Exploits': 1,
    'Backdoor': 8,
    'Worms': 7,
    'Reconnaissance': 2,
    'Shellcode': 5
}

fileCIC['attack_cat'] = fileCIC['attack_cat'].replace(attackEncoder)

columnsToNormalise = [col for col in fileCIC.columns if col != 'attack_cat']

fileCIC[columnsToNormalise] = fileCIC[columnsToNormalise].apply(pd.to_numeric, errors='coerce')
fileCIC.replace([np.inf, -np.inf], np.nan, inplace=True)
fileCIC.dropna(inplace=True)

fileCIC[columnsToNormalise] = MinMaxScaler().fit_transform(fileCIC[columnsToNormalise])

print(fileCIC.shape)
print(fileCIC['attack_cat'].unique())

fileCIC.to_csv('Data for ML/CIC/CIC_DataFrame.csv', index=False)
print("Data Preped")