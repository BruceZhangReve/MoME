import pandas as pd
import numpy as np
import json
import os

# =========================
# Configuration
# =========================
INPUT_LEN = 14  # number of time steps in input
OUTPUT_LEN = 3  # number of time steps in output
TEST_RATIO = 0.2
RESOLUTION_HOURS = 3  # time step resolution

TS_PATH = 'EPA-Air/time_series.csv'
TEXT_PATH = 'EPA-Air/text.csv'
OUT_DIR = 'processed/EPA-Air'

# Feature columns
FEATURE_COLS = [
    'temp',
    'pm2_5',
    'aqi',
    'ozone'
]

# Aggregation rules: all sensor readings -> mean
AGG_RULES = {
    'temp': 'mean',
    'pm2_5': 'mean',
    'aqi': 'mean',
    'ozone': 'mean'
}

# =========================
# Read data
# =========================
print("Reading raw data...")
df_ts = pd.read_csv(TS_PATH, parse_dates=['date_time'])
df_text = pd.read_csv(TEXT_PATH, parse_dates=['date_time'])

# =========================
# Create time bins
# =========================
print(f"Aggregating to {RESOLUTION_HOURS}-hour bins...")

# Round down to nearest RESOLUTION_HOURS
def round_to_bin(dt):
    hours = dt.hour // RESOLUTION_HOURS * RESOLUTION_HOURS
    return dt.replace(hour=hours, minute=0, second=0, microsecond=0)

df_ts['time_bin'] = df_ts['date_time'].apply(round_to_bin)
df_text['time_bin'] = df_text['date_time'].apply(round_to_bin)

# Aggregate by time bin
agg_features = df_ts.groupby('time_bin').agg(AGG_RULES).reset_index()
agg_features = agg_features.sort_values('time_bin').reset_index(drop=True)

# Rename to 'date' for compatibility
agg_features = agg_features.rename(columns={'time_bin': 'date'})
agg_features['date'] = agg_features['date'].dt.date

print(f"After aggregation: {len(agg_features)} time bins ({RESOLUTION_HOURS}h each)")
print(f"Time range: {agg_features['date'].min()} to {agg_features['date'].max()}")

# =========================
# Process text: latest text per bin that is before or in current bin
# =========================
print("Processing text data...")
df_text = df_text.rename(columns={'text': 'summary'})
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

    # Get candidate text: within input window only
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
    f.write(f"EPA-Air {RESOLUTION_HOURS}-hour Aggregated Dataset\n")
    f.write(f"================================================\n")
    f.write(f"Date range: {agg_features['date'].min()} to {agg_features['date'].max()}\n")
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
