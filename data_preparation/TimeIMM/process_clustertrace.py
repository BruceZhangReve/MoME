import pandas as pd
import numpy as np
import json
import os
from datetime import timedelta

# =========================
# Configuration
# =========================
INPUT_LEN = 7  # number of time steps in input
OUTPUT_LEN = 3  # number of time steps in output
TEST_RATIO = 0.2
RESOLUTION_HOURS = 3  # time step resolution

TS_PATH = 'ClusterTrace/time_series.csv'
TEXT_PATH = 'ClusterTrace/text.csv'
OUT_DIR = 'processed/ClusterTrace'

# Feature columns and aggregation methods
FEATURE_COLS = [
    'n_inst',
    'n_task',
    'n_job',
    'machine_cpu_iowait',
    'machine_cpu_kernel',
    'machine_cpu_usr',
    'machine_gpu',
    'machine_load_1',
    'machine_net_receive',
    'machine_num_worker',
    'machine_cpu'
]

AGG_RULES = {
    'n_inst': 'sum',
    'n_task': 'sum',
    'n_job': 'sum',
    'machine_cpu_iowait': 'mean',
    'machine_cpu_kernel': 'mean',
    'machine_cpu_usr': 'mean',
    'machine_gpu': 'mean',
    'machine_load_1': 'mean',
    'machine_net_receive': 'mean',
    'machine_num_worker': 'mean',
    'machine_cpu': 'mean'
}

# =========================
# Read data
# =========================
print("Reading raw data...")
df_ts = pd.read_csv(TS_PATH)
df_text = pd.read_csv(TEXT_PATH)

# Convert date_time to datetime
df_ts['date_time'] = pd.to_datetime(df_ts['date_time'])
df_text['date_time'] = pd.to_datetime(df_text['date_time'])

# =========================
# Create time bins with RESOLUTION_HOURS
# =========================
print(f"Aggregating to {RESOLUTION_HOURS}-hour bins...")

# Round down to nearest RESOLUTION_HOURS
def round_to_bin(dt):
    hours = dt.hour // RESOLUTION_HOURS * RESOLUTION_HOURS
    return dt.replace(hour=hours, minute=0, second=0, microsecond=0)

df_ts['time_bin'] = df_ts['date_time'].apply(round_to_bin)
df_text['time_bin'] = df_text['date_time'].apply(round_to_bin)

# Drop rows where all feature values are NaN
df_ts_valid = df_ts.dropna(subset=FEATURE_COLS, how='all')
print(f"Original: {len(df_ts)} rows, after dropping empty: {len(df_ts_valid)} rows")

# Aggregate by time bin
agg_features = df_ts_valid.groupby('time_bin').agg(AGG_RULES).reset_index()
agg_features = agg_features.sort_values('time_bin').reset_index(drop=True)

# Rename to 'date' for consistency
agg_features = agg_features.rename(columns={'time_bin': 'date'})
agg_features['date'] = agg_features['date'].dt.date  # actually time_bin, keep name for compatibility

print(f"After aggregation: {len(agg_features)} time bins ({RESOLUTION_HOURS}h each)")
print(f"Time range: {agg_features['date'].min()} to {agg_features['date'].max()}")

# =========================
# Process text: latest text per bin that is before or in current bin
# =========================
print("Processing text data...")
df_text['summary'] = df_text['summary'].fillna('').astype(str).str.strip()
df_text = df_text[df_text['summary'] != '']
df_text = df_text.sort_values('date_time')

# Keep latest text per time_bin
latest_text_per_bin = (
    df_text.groupby('time_bin', as_index=False)
    .agg({'date_time': 'max', 'summary': 'last'})
    .sort_values('time_bin')
)
# Convert date to date object for comparison
latest_text_per_bin['date'] = latest_text_per_bin['time_bin'].dt.date

print(f"Text data: {len(df_text)} entries, {len(latest_text_per_bin)} bins with text")

# =========================
# Helper
# =========================
def build_channel_dict(data):
    out = {}
    for j, col in enumerate(FEATURE_COLS):
        out[col] = [float(x) if not np.isnan(x) else 0.0 for x in data[:, j]]
    return out

# =========================
# Generate samples
# =========================
print("Generating sliding window samples...")
samples = []
dropped = 0

total_bins = len(agg_features)

for i in range(total_bins - INPUT_LEN - OUTPUT_LEN + 1):
    input_slice = agg_features.iloc[i:i + INPUT_LEN]
    output_slice = agg_features.iloc[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN]

    input_bins = input_slice['date'].tolist()
    output_bins = output_slice['date'].tolist()

    input_data = input_slice[FEATURE_COLS].values
    output_data = output_slice[FEATURE_COLS].values

    input_start_bin = input_bins[0]
    output_start_bin = output_bins[0]

    # Get candidate text: any text in bins >= input_start and < output_start (i.e. within input window)
    candidate = latest_text_per_bin[
        (latest_text_per_bin['date'] >= input_start_bin) &
        (latest_text_per_bin['date'] < output_start_bin)
    ]

    if len(candidate) == 0:
        dropped += 1
        continue

    # take latest candidate
    chosen = candidate.iloc[-1]

    sample = {
        "text": chosen['summary'],
        "text_timestamp": str(chosen['date_time']),

        "input_timestamps": json.dumps([str(d) for d in input_bins]),
        "output_timestamps": json.dumps([str(d) for d in output_bins]),

        "input_window": json.dumps(build_channel_dict(input_data)),
        "output_window": json.dumps(build_channel_dict(output_data)),
    }

    samples.append(sample)

print(f"Kept: {len(samples)}, Dropped: {dropped}")

# =========================
# Split by time (no shuffle, keep temporal order)
# =========================
n = len(samples)
n_test = int(n * TEST_RATIO)

train_samples = samples[:n - n_test]
test_samples = samples[n - n_test:]

print(f"Split (80%/20%): train={len(train_samples)}, test={len(test_samples)}")

# =========================
# Save CSV
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

pd.DataFrame(train_samples).to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
pd.DataFrame(test_samples).to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)

# Save info
with open(os.path.join(OUT_DIR, 'info.txt'), 'w') as f:
    f.write(f"ClusterTrace {RESOLUTION_HOURS}-hour Aggregated Dataset\n")
    f.write(f"==================================================\n")
    f.write(f"Time range: {agg_features['date'].min()} to {agg_features['date'].max()}\n")
    f.write(f"Resolution: {RESOLUTION_HOURS} hours per time step\n")
    f.write(f"Total time steps: {len(agg_features)}\n")
    f.write(f"Input length: {INPUT_LEN} steps\n")
    f.write(f"Output length: {OUTPUT_LEN} steps\n")
    f.write(f"Total samples: {len(samples)}\n")
    f.write(f"Train samples: {len(train_samples)} ({(1-TEST_RATIO)*100:.0f}%)\n")
    f.write(f"Test samples: {len(test_samples)} ({TEST_RATIO*100:.0f}%)\n")
    f.write(f"\nFeatures ({len(FEATURE_COLS)} dimensions per time step:\n")
    for col in FEATURE_COLS:
        agg = AGG_RULES[col]
        f.write(f"  - {col}: {agg} per time bin\n")
    f.write(f"\nCSV format:\n")
    f.write(f"  - text: latest text summary within input window (one text per sample)\n")
    f.write(f"  - text_timestamp: timestamp of the selected text\n")
    f.write(f"  - input_timestamps/output_timestamps: JSON list of dates for each step\n")
    f.write(f"  - input_window/output_window: JSON dict where keys are feature names, values are list of values\n")

print(f"\nDone! Output saved to {OUT_DIR}/")
