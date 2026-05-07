import pandas as pd
import numpy as np
import json
import os

# =========================
# Configuration
# =========================
INPUT_LEN = 14  # 14 weeks input
OUTPUT_LEN = 3  # predict next 3 weeks
TEST_RATIO = 0.2

TS_PATH = 'ILINet/time_series.csv'
TEXT_PATH = 'ILINet/text.csv'
OUT_DIR = 'processed/ILINet'

# Feature columns (all except date_time, record_id)
FEATURE_COLS = [
    '% WEIGHTED ILI',
    'ILI %',
    'AGE 0-4',
    'AGE 25-49',
    'AGE 25-64',
    'AGE 5-24',
    'AGE 50-64',
    'AGE 65',
    'ILITOTAL',
    'NUM. OF PROVIDERS',
    'TOTAL PATIENTS'
]

AGG_RULES = {col: 'mean' for col in FEATURE_COLS}

# =========================
# Read data
# =========================
print("Reading raw data...")
df_ts = pd.read_csv(TS_PATH, parse_dates=['date_time'])
df_text = pd.read_csv(TEXT_PATH, parse_dates=['date_time'])

# Extract date (week)
df_ts['date'] = df_ts['date_time'].dt.date
df_text['date'] = df_text['date_time'].dt.date

# Already weekly, no need to aggregate, just sort
weekly_features = df_ts.sort_values('date_time').reset_index(drop=True)

print(f"Total weeks: {len(weekly_features)}")
print(f"Date range: {weekly_features['date'].min()} to {weekly_features['date'].max()}")

# =========================
# Clean text
# =========================
print("Processing text data...")
df_text = df_text.rename(columns={'text': 'summary'})
df_text['summary'] = df_text['summary'].fillna('').astype(str).str.strip()
df_text = df_text[df_text['summary'] != '']
df_text = df_text.sort_values('date_time')

# Keep latest text per week
latest_text_per_week = (
    df_text.groupby('date', as_index=False)
    .agg({'date_time': 'max', 'summary': 'last'})
    .sort_values('date')
)

print(f"Text data: {len(df_text)} entries, {len(latest_text_per_week)} weeks with text")

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

total_weeks = len(weekly_features)

for i in range(total_weeks - INPUT_LEN - OUTPUT_LEN + 1):
    input_slice = weekly_features.iloc[i:i + INPUT_LEN]
    output_slice = weekly_features.iloc[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN]

    input_dates = input_slice['date'].tolist()
    output_dates = output_slice['date'].tolist()

    input_data = input_slice[FEATURE_COLS].values
    output_data = output_slice[FEATURE_COLS].values

    input_start = input_dates[0]
    pred_start = output_dates[0]

    # select latest feasible text (within input window, before prediction starts)
    candidate = latest_text_per_week[
        (latest_text_per_week['date'] >= input_start) &
        (latest_text_per_week['date'] < pred_start)
    ]

    if len(candidate) == 0:
        dropped += 1
        continue

    chosen = candidate.iloc[-1]

    sample = {
        "text": chosen['summary'],
        "text_timestamp": str(chosen['date_time']),

        "input_timestamps": json.dumps([str(d) for d in input_dates]),
        "output_timestamps": json.dumps([str(d) for d in output_dates]),

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
    f.write(f"ILINet Weekly Dataset (already weekly in raw data)\n")
    f.write(f"===============================================\n")
    f.write(f"Date range: {weekly_features['date'].min()} to {weekly_features['date'].max()}\n")
    f.write(f"Resolution: 1 week per time step\n")
    f.write(f"Total weeks: {len(weekly_features)}\n")
    f.write(f"Input length: {INPUT_LEN} weeks\n")
    f.write(f"Output length: {OUTPUT_LEN} weeks\n")
    f.write(f"Total samples: {len(samples)}\n")
    f.write(f"Train samples: {len(train_samples)} ({(1-TEST_RATIO)*100:.0f}%)\n")
    f.write(f"Test samples: {len(test_samples)} ({TEST_RATIO*100:.0f}%)\n")
    f.write(f"\nFeatures ({len(FEATURE_COLS)} dimensions per time step:\n")
    for col in FEATURE_COLS:
        f.write(f"  - {col}\n")
    f.write(f"\nCSV format:\n")
    f.write(f"  - text: latest text summary within input window (one text per sample)\n")
    f.write(f"  - text_timestamp: timestamp of the selected text\n")
    f.write(f"  - input_timestamps/output_timestamps: JSON list of dates for each step\n")
    f.write(f"  - input_window/output_window: JSON dict where keys are feature names, values are list of values\n")

print("\nDone! Output saved to processed/ILINet/")
