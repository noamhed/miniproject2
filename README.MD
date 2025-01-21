# README: ECoG Data ERP Analysis

## Project Overview
This project processes electrophysiological data (ECoG) to calculate the Event-Related Potential (ERP) for different trials associated with specific finger movements. By averaging the brain signals over multiple trials for each finger, this analysis aims to extract time-locked patterns of neural activity.

## Files in the Project
- **`calc_mean_erp.py`**: Contains the Python function `calc_mean_erp` which processes ECoG data to compute mean ERP values for each finger.
- **`events_file_ordered.csv`**: A CSV file listing the trial data, including start times, peaks, and finger associations.
- **`brain_data_channel_one.csv`**: A CSV file containing ECoG data from a single channel, recorded as a time series.

## Dependencies
To run this project, you need the following libraries installed:
- `pandas`
- `numpy`

You can install them using:
```bash
pip install pandas numpy
```

## Function Explanation
### `calc_mean_erp(trial_points: pd.DataFrame, ecog_data: np.ndarray) -> np.ndarray`
This function calculates the mean ERP for each finger by processing the ECoG data within specified time windows.

### Inputs:
1. `trial_points`:
   - Columns: Start time, Peak time, and Finger identifier.
   - Data should be integers.
2. `ecog_data`:
   - A 1D array representing the time-series ECoG data from a single channel.

### Outputs:
- `fingers_erp_mean` (np.ndarray):
   - A 2D array of shape `(5, 1201)`.
   - Each row corresponds to a finger (1–5).
   - Columns represent averaged signal values over the 1201 time points (-200 ms to +1000 ms relative to trial start).
