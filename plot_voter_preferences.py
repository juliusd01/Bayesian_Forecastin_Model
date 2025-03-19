import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime

# Directory containing the CSV files
csv_dir = "/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/Final/trends"

# Get all CSV files sorted lexicographically
csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))

# Initialize a dictionary to store mean values for each party (0-7)
party_means = {party: [] for party in range(8)}

# Process each CSV file
for file in csv_files:
    df = pd.read_csv(file, index_col=0)  # First column as index
    df = df[8:]  # Assuming you want to skip the first 8 rows
    for idx in df.index:
        # Extract party number from the index (e.g., 'voter_share_nation[0]' -> 0)
        party = int(idx.split('[')[1].split(']')[0])
        mean_value = df.loc[idx, 'mean']
        party_means[party].append(mean_value)

# Map party numbers to party names
party_names = {
    0: "CDU",
    1: "SPD",
    2: "Greens",
    3: "FDP",
    4: "Left",
    5: "AfD",
    6: "BSW",
    7: "Other"
}

# Rename the keys in the data
party_means = {party_names[key]: value for key, value in party_means.items()}

# Define the custom date list
date_list = [
    datetime(2024, 11, 13), datetime(2024, 11, 23), datetime(2024, 12, 3),
    datetime(2024, 12, 13), datetime(2024, 12, 23), datetime(2025, 1, 3),
    datetime(2025, 1, 13), datetime(2025, 1, 23), datetime(2025, 2, 3),
    datetime(2025, 2, 13), datetime(2025, 2, 23)
]

# Plotting
plt.figure(figsize=(12, 6))
for party in party_means:
    plt.plot(date_list, party_means[party], marker='o', linestyle='-', label=f'{party}')

# Set x-axis ticks and labels explicitly
plt.xticks(date_list, [date.strftime('%Y-%m-%d') for date in date_list], rotation=45)

plt.xlabel('Time')
plt.ylabel('Mean Voter Share')
plt.title('Mean Voter Share Over Time by Party')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("/home/juliusd/documents/Quant-Econ/Bayesian_Modelling/Term_paper/results/voter_preferences.png")
plt.show()