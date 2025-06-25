import pandas as pd
import numpy as np

# --- 1. Predefined Mappings and Configuration ---

# This mapping converts team abbreviations to unique integer IDs, inspired by your example.
TEAM_TO_ID = {
    'ARI': 1, 'ATL': 2, 'BAL': 3, 'BUF': 4, 'CAR': 5, 'CHI': 6, 'CIN': 7,
    'CLE': 8, 'DAL': 9, 'DEN': 10, 'DET': 11, 'GB': 12, 'HOU': 13, 'IND': 14,
    'JAX': 15, 'KC': 16, 'MIA': 20, 'MIN': 21, 'NE': 22, 'NO': 23, 'NYG': 24,
    'NYJ': 25, 'PHI': 26, 'PIT': 27, 'SF': 28, 'SEA': 29, 'TB': 30, 'TEN': 31,
    'WAS': 32, 'SD': 33, 'STL': 34, 'OAK': 35, 'JAC': 15 # Aliases for 2015 data
}

INPUT_FILENAME = 'NFLPlaybyPlay2015.csv'
OUTPUT_FILENAME = 'qml_real_data_eagles_2015.csv'
TEAM_CODE = 'PHI'

print("--- Starting Data Preparation with Interpretable Data ---")

# --- 2. Load and Initial Filtering ---
try:
    df = pd.read_csv(INPUT_FILENAME, low_memory=False)
    print(f"Successfully loaded '{INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"FATAL ERROR: The input file '{INPUT_FILENAME}' was not found.")
    exit()

# Filter for games involving the specified team
df_team = df[(df['posteam'] == TEAM_CODE) | (df['DefensiveTeam'] == TEAM_CODE)].copy()
print(f"Filtered dataset to plays involving '{TEAM_CODE}'.")

# Filter for only 'Run' and 'Pass' plays, which are our focus
df_filtered = df_team[df_team['PlayType'].isin(['Run', 'Pass'])].copy()
print(f"Filtered for 'Run' and 'Pass' plays, resulting in {len(df_filtered)} samples.")


# --- 3. Feature Selection and Cleaning ---

# Select the raw features we need for our model
features = [
    'qtr',
    'down',
    'TimeSecs',
    'yrdline100',
    'ydstogo',
    'ScoreDiff',
    'posteam',
    'DefensiveTeam',
    'PlayType'
]
df_clean = df_filtered[features].copy()

# Handle missing values for key numeric columns by filling with the median
for col in ['down', 'TimeSecs', 'yrdline100', 'ScoreDiff', 'ydstogo']:
    if df_clean[col].isnull().any():
        median_val = df_clean[col].median()
        # Use .loc to safely fill values on the DataFrame
        df_clean.loc[:, col] = df_clean[col].fillna(median_val)
        print(f"Filled missing values in '{col}' with median: {median_val}")

# Drop any rows that might still have missing data
df_clean.dropna(inplace=True)
print(f"Dataset size after cleaning: {len(df_clean)} plays.")

# --- 4. Feature Engineering and Final Formatting ---

# Convert team names to numeric IDs using the map
df_clean['offense_team_id'] = df_clean['posteam'].map(TEAM_TO_ID)
df_clean['defense_team_id'] = df_clean['DefensiveTeam'].map(TEAM_TO_ID)

# Encode the target variable 'PlayType' into a binary numeric format
# 'Pass' -> 1
# 'Run'  -> 0
df_clean['play_type_numeric'] = df_clean['PlayType'].apply(lambda x: 1 if x == 'Pass' else 0)

# Convert float columns that should be integers (like 'down')
for col in ['qtr', 'down', 'TimeSecs', 'yrdline100', 'ydstogo', 'ScoreDiff']:
    # Fill any potential new NaNs from mapping before converting to int
    if df_clean[col].isnull().any():
        df_clean.loc[:, col] = df_clean[col].fillna(0)
    df_clean[col] = df_clean[col].astype(int)

print("Converted feature columns to integer types for clarity.")

# --- 5. Final Column Selection and Save ---

# Define the final set of columns for the QML model. All data is now numeric and interpretable.
final_columns = [
    'qtr',
    'down',
    'TimeSecs',
    'yrdline100',
    'ydstogo',
    'ScoreDiff',
    'offense_team_id',
    'defense_team_id',
    'play_type_numeric' # This is our target variable (y)
]

final_df = df_clean[final_columns].copy()

# Final check for any missing values created during mapping
final_df.dropna(inplace=True)

# Save the fully processed, "real data" file
final_df.to_csv(OUTPUT_FILENAME, index=False)
print(f"\n--- Process Complete! ---")
print(f"QML-ready data with real, interpretable values saved to '{OUTPUT_FILENAME}'.")

print("\n--- Final Data Preview (First 10 Rows) ---")
print(final_df.head(10))
