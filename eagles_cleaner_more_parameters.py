import pandas as pd
import numpy as np

# --- 1. Configuration and Team Mapping ---
TEAM_TO_ID = {
    'ARI': 1, 'ATL': 2, 'BAL': 3, 'BUF': 4, 'CAR': 5, 'CHI': 6, 'CIN': 7,
    'CLE': 8, 'DAL': 9, 'DEN': 10, 'DET': 11, 'GB': 12, 'HOU': 13, 'IND': 14,
    'JAX': 15, 'KC': 16, 'MIA': 20, 'MIN': 21, 'NE': 22, 'NO': 23, 'NYG': 24,
    'NYJ': 25, 'PHI': 26, 'PIT': 27, 'SF': 28, 'SEA': 29, 'TB': 30, 'TEN': 31,
    'WAS': 32, 'SD': 33, 'STL': 34, 'OAK': 35, 'JAC': 15
}

INPUT_FILENAME = 'NFLPlaybyPlay2015.csv'
OUTPUT_FILENAME = 'qml_eagles_2015_enhanced.csv'
TEAM_CODE = 'PHI'

print("--- Starting Enhanced Data Preparation for Eagles ---")

# --- 2. Load and Initial Filtering ---
try:
    df = pd.read_csv(INPUT_FILENAME, low_memory=False)
    print(f"Loaded '{INPUT_FILENAME}' with {len(df)} records.")
except FileNotFoundError:
    print(f"ERROR: '{INPUT_FILENAME}' not found.")
    exit()

# Filter for plays where Eagles are the offense
df_eagles = df[df['posteam'] == TEAM_CODE].copy()
print(f"Filtered for Eagles as offense: {len(df_eagles)} plays.")

# Focus on 'Run' and 'Pass' plays
df_filtered = df_eagles[df_eagles['PlayType'].isin(['Run', 'Pass'])].copy()
print(f"Selected 'Run' and 'Pass' plays: {len(df_filtered)} plays.")

# --- 3. Feature Engineering ---

# Initialize feature DataFrame with core features
features = [
    'GameID', 'qtr', 'down', 'TimeSecs', 'yrdline100', 'ydstogo', 'ScoreDiff',
    'PlayType', 'Yards.Gained', 'posteam', 'DefensiveTeam'
]
df_features = df_filtered[features].copy()

# Handle missing values for numeric columns
numeric_cols = ['down', 'TimeSecs', 'yrdline100', 'ydstogo', 'ScoreDiff', 'Yards.Gained']
for col in numeric_cols:
    if df_features[col].isnull().any():
        median_val = df_features[col].median()
        df_features.loc[:, col] = df_features[col].fillna(median_val)
        print(f"Filled missing '{col}' with median: {median_val}")

# Drop rows with any remaining missing values
df_features.dropna(inplace=True)
print(f"After cleaning: {len(df_features)} plays.")

# Calculate in-game passing proportion
def calculate_pass_proportion(group):
    group['pass_count'] = (group['PlayType'] == 'Pass').cumsum()
    group['total_plays'] = np.arange(1, len(group) + 1)
    group['pass_proportion'] = group['pass_count'] / group['total_plays']
    return group

df_features = df_features.groupby('GameID').apply(calculate_pass_proportion).reset_index(drop=True)
df_features['pass_proportion'] = df_features['pass_proportion'].fillna(0)

# Calculate average yards gained on previous plays (pass and run separately)
df_features['prev_pass_yards'] = df_features.groupby('GameID').apply(
    lambda x: x['Yards.Gained'].where(x['PlayType'] == 'Pass').shift(1).rolling(window=5, min_periods=1).mean()
).reset_index(level=0, drop=True).fillna(0)
df_features['prev_run_yards'] = df_features.groupby('GameID').apply(
    lambda x: x['Yards.Gained'].where(x['PlayType'] == 'Run').shift(1).rolling(window=5, min_periods=1).mean()
).reset_index(level=0, drop=True).fillna(0)

# Calculate consecutive play types (number of same play types before current play)
def count_consecutive_plays(group):
    group['consecutive_plays'] = (group['PlayType'] != group['PlayType'].shift(1)).cumsum()
    group['consecutive_count'] = group.groupby('consecutive_plays').cumcount()
    return group

df_features = df_features.groupby('GameID').apply(count_consecutive_plays).reset_index(drop=True)
df_features['consecutive_count'] = df_features['consecutive_count'].astype(int)

# Convert team IDs to numeric
df_features['defense_team_id'] = df_features['DefensiveTeam'].map(TEAM_TO_ID).fillna(0).astype(int)

# Encode PlayType: Pass=1, Run=0
df_features['play_type_numeric'] = (df_features['PlayType'] == 'Pass').astype(int)

# Convert float columns to integers
int_cols = ['qtr', 'down', 'TimeSecs', 'yrdline100', 'ydstogo', 'ScoreDiff', 'consecutive_count']
for col in int_cols:
    df_features[col] = df_features[col].astype(int)

print("Engineered features: pass_proportion, prev_pass_yards, prev_run_yards, consecutive_count.")

# --- 4. Feature Selection ---
# Select high-importance features based on classical study
final_columns = [
    'qtr', 'down', 'yrdline100', 'ydstogo', 'ScoreDiff',
    'pass_proportion', 'prev_pass_yards', 'prev_run_yards', 'consecutive_count',
    'defense_team_id', 'play_type_numeric'
]

final_df = df_features[final_columns].copy()
final_df.dropna(inplace=True)
print(f"Final dataset: {len(final_df)} plays with {len(final_columns)} features.")

# --- 5. Save Output ---
final_df.to_csv(OUTPUT_FILENAME, index=False)
print(f"\n--- Completed! Saved to '{OUTPUT_FILENAME}' ---")
print("\n--- Data Preview (First 5 Rows) ---")
print(final_df.head(5))