import pandas as pd

# Load the CSV file
df = pd.read_csv("Big Ten.csv")

# Filter for Ohio State offensive plays
ohio_state_offense = df[df["offense_play"] == "Ohio State"]

# Save the filtered data to a .qsv file
ohio_state_offense.to_csv("qml_ohiostate_enhanced.csv", index=False)