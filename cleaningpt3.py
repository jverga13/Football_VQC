import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- 1. Predefined Mappings and Constants ---

TEAM_TO_ID = {
    'ARI': 1, 'ATL': 2, 'BAL': 3, 'BUF': 4, 'CAR': 5, 'CHI': 6, 'CIN': 7,
    'CLE': 8, 'DAL': 9, 'DEN': 10, 'DET': 11, 'GB': 12, 'HOU': 13, 'IND': 14,
    'JAX': 15, 'KC': 16, 'LV': 17, 'LAC': 18, 'LAR': 19, 'MIA': 20, 'MIN': 21,
    'NE': 22, 'NO': 23, 'NYG': 24, 'NYJ': 25, 'PHI': 26, 'PIT': 27, 'SF': 28,
    'SEA': 29, 'TB': 30, 'TEN': 31, 'WAS': 32
}

MASCOT_TO_ABBREVIATION = {
    '49ers': 'SF', 'Bears': 'CHI', 'Bengals': 'CIN', 'Bills': 'BUF', 'Broncos': 'DEN',
    'Browns': 'CLE', 'Buccaneers': 'TB', 'Cardinals': 'ARI', 'Chargers': 'LAC',
    'Chiefs': 'KC', 'Colts': 'IND', 'Commanders': 'WAS', 'Cowboys': 'DAL',
    'Dolphins': 'MIA', 'Eagles': 'PHI', 'Falcons': 'ATL', 'Giants': 'NYG',
    'Jaguars': 'JAX', 'Jets': 'NYJ', 'Lions': 'DET', 'Packers': 'GB',
    'Panthers': 'CAR', 'Patriots': 'NE', 'Raiders': 'LV', 'Rams': 'LAR', 'LA': 'LAR',
    'Ravens': 'BAL', 'Saints': 'NO', 'Seahawks': 'SEA', 'Steelers': 'PIT',
    'Texans': 'HOU', 'Titans': 'TEN', 'Vikings': 'MIN'
}

TEAM_NAME_CORRECTIONS = {
    'JAC': 'JAX', 'WSH': 'WAS', 'STL': 'LAR', 'SD': 'LAC'
}

DAY_TO_NUM = {'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 'FRI': 4, 'SAT': 5, 'SUN': 6}

# --- 2. Helper Functions for Data Transformation ---

def assign_week_order(week):
    """Assigns numerical order to weeks for proper chronological sorting."""
    if pd.isna(week):
        return 0
    week = str(week).lower()
    if 'hall of fame' in week:
        return -2
    if 'preseason' in week:
        return -1
    if 'super bowl' in week:
        return 100
    if 'conference championships' in week:
        return 99
    if 'divisional playoffs' in week:
        return 98
    if 'wild card' in week:
        return 97
    try:
        return float(week)
    except ValueError:
        return 0

def calculate_score_differential(df):
    """
    Calculates running score differential for each play, using IsScoringPlay and PlayDescription.
    """
    df['home_score'] = 0
    df['away_score'] = 0
    df['score_differential'] = 0

    def apply_score(game_df):
        home_score = 0
        away_score = 0
        for index, play in game_df.iterrows():
            possession_team_score = home_score if play['TeamWithPossession'] == play['HomeTeam'] else away_score
            opponent_team_score = away_score if play['TeamWithPossession'] == play['HomeTeam'] else home_score
            game_df.loc[index, 'score_differential'] = possession_team_score - opponent_team_score

            if play.get('IsScoringPlay', 0) == 1:
                play_desc = str(play.get('PlayDescription', '')).lower()
                points = 0
                team_that_scored = play['TeamWithPossession']

                if 'touchdown' in play_desc:
                    points = 6
                elif 'extra point is good' in play_desc:
                    points = 1
                elif 'two-point conversion attempt' in play_desc and 'succeeds' in play_desc:
                    points = 2
                elif 'field goal is good' in play_desc:
                    points = 3
                elif 'safety' in play_desc:
                    points = 2
                    team_that_scored = play['HomeTeam'] if play['TeamWithPossession'] == play['AwayTeam'] else play['AwayTeam']

                if points > 0:
                    if team_that_scored == play['HomeTeam']:
                        home_score += points
                    else:
                        away_score += points

        return game_df

    df = df.groupby('game_id').apply(apply_score, include_groups=False).reset_index()
    return df

def preprocess_nfl_for_qvc(df, include_all_games=True):
    """
    Parses and transforms the raw NFL DataFrame into a fully numeric format.
    """
    df = df.copy()

    # Step A: Clean and Standardize Team Names
    team_cols = ['TeamWithPossession', 'AwayTeam', 'HomeTeam']
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().replace(MASCOT_TO_ABBREVIATION).replace(TEAM_NAME_CORRECTIONS)

    # Step B: Initial Time Parsing and Date Handling
    df['quarter'] = df['Quarter'].str.extract(r'(\d+)|(\bOT\b)', expand=True)[0].fillna(df['Quarter'].str.contains('OT', case=False, na=False).astype(int) * 5).astype(float)
    time_extract = df['PlayDescription'].str.extract(r'\((\d{1,2}):(\d{2})\)')
    time_extract.columns = ['minutes', 'seconds']
    time_extract = time_extract.apply(pd.to_numeric, errors='coerce')
    df['time_remaining_in_quarter'] = time_extract['minutes'] * 60 + time_extract['seconds']
    
    # Parse dates and handle invalid ones
    df['full_date'] = pd.to_datetime(df['Date'] + '/' + df['Season'].astype(str), format='%m/%d/%Y', errors='coerce')
    invalid_dates = df['full_date'].isna()
    if invalid_dates.any():
        print(f"WARNING: {invalid_dates.sum()} invalid dates found. Attempting to fill with fallback parsing.")
        df.loc[invalid_dates, 'full_date'] = pd.to_datetime(df.loc[invalid_dates, 'Date'], errors='coerce', format='%m/%d')
        df.loc[invalid_dates, 'full_date'] = df.loc[invalid_dates, 'full_date'].apply(
            lambda x: x.replace(year=2024) if pd.notna(x) else pd.Timestamp('2024-01-01')
        )
    df['day_of_year'] = df['full_date'].dt.dayofyear
    df['Week_Order'] = df['Week'].apply(assign_week_order)

    # Step C: Create Game ID and Sort Chronologically
    df['game_id'] = df['full_date'].dt.strftime('%Y%m%d') + '_' + df['AwayTeam'] + '@' + df['HomeTeam']
    df.sort_values(
        by=['full_date', 'Week_Order', 'game_id', 'quarter', 'time_remaining_in_quarter'],
        ascending=[True, True, True, True, False],
        inplace=True
    )
    df.reset_index(drop=True, inplace=True)
    print("--- Data has been chronologically sorted by Date and Week. ---")

    # Step D: Calculate Running Score Differential
    df = calculate_score_differential(df)
    print("--- Running score differential calculated. ---")

    # Step E: Filter out non-action plays
    play_action_keywords = ['Kickoff', 'Punt', 'Field Goal', 'Extra Point', 'Timeout', 'END QUARTER', 'END GAME', 'Two-Minute Warning', 'No Play']
    mask = df['PlayDescription'].str.contains('|'.join(play_action_keywords), case=False, na=False)
    df = df[~mask]
    print(f"--- Non-action plays removed. Rows remaining: {len(df)} ---")

    # Step F: Parse 'PlayStart'
    start_pattern = r'(\d)(?:st|nd|rd|th)\s*&\s*(\d+|inches|goal)[\s\w]*at\s(?:(50|MIDFIELD)|([A-Z]{2,3})\s(\d{1,2}))'
    play_start_extract = df['PlayStart'].str.extract(start_pattern, re.IGNORECASE)
    play_start_extract.columns = ['down', 'distance', 'midfield', 'side_of_field', 'yard_line_num']
    df[['down', 'distance', 'midfield', 'side_of_field', 'yard_line_num']] = play_start_extract

    # Debugging step for PlayStart parsing
    print("\n--- DEBUG: Checking for PlayStart parsing failures ---")
    failed_parses = df['down'].isnull().sum()
    if failed_parses > 0:
        print(f"WARNING: The 'PlayStart' pattern failed to parse {failed_parses} rows.")
    else:
        print("PlayStart parsing was successful for all remaining rows.")
    print("------------------------------------------------------\n")

    # Step G: Final Feature Engineering
    df['down'] = pd.to_numeric(df['down'], errors='coerce')
    df['distance_to_first'] = df['distance'].replace({'inches': 0.5, 'goal': 1.0}).astype(float)
    df['yard_line_num'] = pd.to_numeric(df['yard_line_num'], errors='coerce')
    df['field_position_numeric'] = np.where(df['side_of_field'] == df['TeamWithPossession'], 100 - df['yard_line_num'], df['yard_line_num'])
    df.loc[df['midfield'].notna(), 'field_position_numeric'] = 50.0
    pass_keywords = 'pass|sacked|scrambles|incomplete'
    df['play_type'] = np.where(df['PlayDescription'].str.contains(pass_keywords, case=False, na=False), 1, 0)
    df['play_result_yards'] = df['PlayOutcome'].str.extract(r'(-?\d+)\sYard').astype(float).fillna(0)
    df['is_interception'] = df['PlayOutcome'].str.contains('Interception', na=False).astype(int)
    df['is_fumble'] = (df['PlayOutcome'].str.contains('Fumble', na=False) | df['PlayDescription'].str.contains('fumble', na=False)).astype(int)
    
    # Enhanced time remaining calculation
    df['total_seconds_remaining'] = (4 - df['quarter']) * 900 + df['time_remaining_in_quarter']
    df.loc[df['quarter'] > 4, 'total_seconds_remaining'] = df['time_remaining_in_quarter']  # Overtime
    df.loc[df['quarter'].isna(), 'total_seconds_remaining'] = 0
    df.loc[df['total_seconds_remaining'] < 0, 'total_seconds_remaining'] = 0

    df['DefensiveTeam'] = np.where(df['TeamWithPossession'] == df['AwayTeam'], df['HomeTeam'], df['AwayTeam'])
    df['offense_team_id'] = df['TeamWithPossession'].map(TEAM_TO_ID)
    df['defense_team_id'] = df['DefensiveTeam'].map(TEAM_TO_ID)
    df['day_num'] = df['Day'].map(DAY_TO_NUM)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_num'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_num'] / 7)

    # Step H: Final Cleanup and Column Selection
    final_feature_columns = [
        'game_id', 'down', 'distance_to_first', 'field_position_numeric', 'play_type',
        'quarter', 'total_seconds_remaining', 'score_differential', 'offense_team_id',
        'defense_team_id', 'day_of_year', 'day_of_week_sin', 'day_of_week_cos',
        'play_result_yards', 'is_interception', 'is_fumble'
    ]
    df_final = df[final_feature_columns].copy()

    # Final Debug check before dropna()
    print("\n--- DEBUG: Checking for missing values BEFORE final dropna() ---")
    nan_counts = df_final.isnull().sum()
    nan_report = nan_counts[nan_counts > 0]
    if not nan_report.empty:
        print("Columns with missing values and their counts:")
        print(nan_report)
    else:
        print("No missing values found in the final feature set. Good to go!")
    print("----------------------------------------------------------\n")
    
    df_final.dropna(inplace=True)
    return df_final

# --- 3. Main Script Execution ---
try:
    raw_df = pd.read_csv('2024_plays.csv', low_memory=False)
    print(f"--- Successfully loaded '2024_plays.csv'. Found {len(raw_df)} rows. ---")

    # Optional filtering (set include_all_games=False to exclude special weeks)
    if not True:  # Set to False to filter out special weeks
        week_filter_keywords = ['Hall of Fame', 'Preseason', 'Super Bowl', 'Conference Championships', 'Divisional Playoffs']
        week_mask = raw_df['Week'].str.contains('|'.join(week_filter_keywords), na=True)
        raw_df = raw_df[~week_mask]
        print(f"--- Preseason and Postseason weeks removed. Rows remaining: {len(raw_df)} ---")

except FileNotFoundError:
    print("Error: '2024_plays.csv' not found. Please place it in the same folder.")
    raw_df = pd.DataFrame()

if not raw_df.empty:
    final_df = preprocess_nfl_for_qvc(raw_df, include_all_games=True)
    
    print(f"\n--- Preprocessing complete. Final dataset has {len(final_df)} rows. ---")

    if not final_df.empty:
        print("\n--- Final Processed DataFrame (first 5 rows) ---")
        print(final_df.head())
    else:
        print("\n--- Final DataFrame is empty. Please check the DEBUG output to see why rows were dropped. ---")

    output_filename = 'qvc_ready_playspt4.csv'
    final_df.to_csv(output_filename, index=False)
    print(f"\n--- Fully numeric, QVC-ready data saved to '{output_filename}' ---")