# Noam Hed 322437237
import pandas as pd
import numpy as np

def calc_mean_erp(trial_points: pd.DataFrame, ecog_data: np.ndarray) -> np.ndarray:
    # Name the columns of the events for easier readability
    trial_points.columns = ["Start", "Peak", "Finger"]
    # Convert all the data in the events csv to the int type
    trial_points = trial_points.astype({'Start': 'int', 'Peak': 'int', 'Finger': 'int'})
    # Create a dictionary to store the ERP data
    erp_data = {1: [], 2: [], 3: [], 4: [], 5: []}
    # Set time frame for each trial to match the brain activity at that time
    for i, row in trial_points.iterrows():  # Loop through each row in the DataFrame
        finger = row['Finger']  # Extract the finger associated with this trial
        original_start_time = row['Start']  # Extract the trial start time
        # Define the adjusted start and end times for the time window
        start_time = original_start_time - 200  # 200 ms before the start time
        end_time = original_start_time + 1000  # 1000 ms after the start time

        # Ensure the time window is within bounds of the ECoG data
        if 0 <= start_time < len(ecog_data) and 0 <= end_time < len(ecog_data):
            # Append the relevant data slice to the corresponding finger
            erp_data[finger].append(ecog_data[start_time:end_time + 1].flatten())  # Ensure 1D array
        else:
            print(f"Skipping trial with out-of-bounds time window: {start_time} to {end_time}")

    # Create an empty matrix with five rows and 1201 columns to contain the averaged data
    fingers_erp_mean = np.zeros((5, 1201))
    # Iterate over each finger and its corresponding list of trials in the erp_data dictionary
    for finger, signals in erp_data.items():  # items() is used so we can separate between finger and signals
        if signals:  # Check if there are any trials recorded for this finger
            # Compute the mean across all trials for the current finger
            fingers_erp_mean[finger - 1, :] = np.mean(signals, axis=0)  # Compute the average for each time point across all trials

    return fingers_erp_mean


# Load the data
trial_points = pd.read_csv("mini_project_2_data/events_file_ordered.csv", header=None)
ecog_data = pd.read_csv("mini_project_2_data/brain_data_channel_one.csv", header=None).values.flatten()

# Call the function with the trial points and ECoG data
print(calc_mean_erp(trial_points, ecog_data))
