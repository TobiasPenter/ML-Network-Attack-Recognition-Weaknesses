import pandas as pd

# Editing file 6 so that there are no non-ascii characters or double labels

file6Path = "Data for ML/CIC/CIC-IDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

df = pd.read_csv(file6Path, encoding="latin-1")

with open(file6Path, "r", encoding="latin-1") as f:
    for i, line in enumerate(f, start=1):
        BruteForceColumns = False

        # Split the line into columns and check for non-ASCII characters
        for col_index, value in enumerate(line.strip().split(",")):
            for char in value:
                if ord(char) > 127:  # Detect non-ASCII characters
                    BruteForceColumns = True
                    break
            if BruteForceColumns:
                break

        if BruteForceColumns:
            # Replace the label with "Brute Force" for the problematic line
            df.at[i - 1, df.columns[-1]] = "Brute Force"

# Save the modified DataFrame with updated labels
df.to_csv("Data for ML/CIC/CIC-IDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_modified.csv", index=False)

# Read the files
file1 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") 
file2 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", header=None)
file3 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv", header=None)
file4 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Monday-WorkingHours.pcap_ISCX.csv", header=None)
file5 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", header=None)
file6 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_modified.csv", header=None)
file7 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Tuesday-WorkingHours.pcap_ISCX.csv", header=None)
file8 = pd.read_csv("Data for ML/CIC/CIC-IDS2017/Wednesday-workingHours.pcap_ISCX.csv", header=None)

columns = file1.columns

file2.columns = columns
file3.columns = columns
file4.columns = columns
file5.columns = columns
file6.columns = columns
file7.columns = columns
file8.columns = columns
# Concatenate the files
data = pd.concat([file1, file2, file3, file4, file5, file6, file7, file8])

data.columns = columns

# Save the new file
data.to_csv("Data for ML/CIC/CIC-IDS2017.csv")