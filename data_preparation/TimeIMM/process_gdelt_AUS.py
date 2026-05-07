import pandas as pd
import os
import json

# =========================
# Configuration
# =========================
INPUT_LEN = 14
OUTPUT_LEN = 3
TEST_RATIO = 0.2

TS_PATH = 'GDELT_AUS/time_series.csv'
TEXT_PATH = 'GDELT_AUS/text.csv'
OUT_DIR = 'processed/GDELT_AUS'

FEATURE_COLS = [
    'GoldsteinScale',
    'NumMentions',
    'NumSources',
    'NumArticles',
    'AvgTone'
]

# =========================
# Read data
# =========================
df_ts = pd.read_csv(TS_PATH)
df_text = pd.read_csv(TEXT_PATH)

df_ts['date_time'] = pd.to_datetime(df_ts['date_time'])
df_text['date_time'] = pd.to_datetime(df_text['date_time'])

df_ts['date'] = df_ts['date_time'].dt.date
df_text['date'] = df_text['date_time'].dt.date

# =========================
# Aggregate TS
# =========================
daily_features = df_ts.groupby('date').agg({
    'GoldsteinScale': 'mean',
    'NumMentions': 'sum',
    'NumSources': 'sum',
    'NumArticles': 'sum',
    'AvgTone': 'mean'
}).reset_index()

daily_features = daily_features.sort_values('date').reset_index(drop=True)

# =========================
# Clean text
# =========================
df_text['summary'] = df_text['summary'].fillna('').astype(str).str.strip()
df_text = df_text[df_text['summary'] != '']
df_text = df_text.sort_values('date_time')

latest_text_per_day = (
    df_text.groupby('date', as_index=False)
    .agg({'date_time': 'max', 'summary': 'last'})
    .sort_values('date')
)

# =========================
# Helper
# =========================
def build_channel_dict(data):
    out = {}
    for j, col in enumerate(FEATURE_COLS):
        out[col] = [float(x) for x in data[:, j]]
    return out

# =========================
# Generate samples
# =========================
samples = []
dropped = 0

total_days = len(daily_features)

for i in range(total_days - INPUT_LEN - OUTPUT_LEN + 1):

    input_slice = daily_features.iloc[i:i + INPUT_LEN]
    output_slice = daily_features.iloc[i + INPUT_LEN:i + INPUT_LEN + OUTPUT_LEN]

    input_dates = input_slice['date'].tolist()
    output_dates = output_slice['date'].tolist()

    input_data = input_slice[FEATURE_COLS].values
    output_data = output_slice[FEATURE_COLS].values

    input_start = input_dates[0]
    pred_start = output_dates[0]

    # select latest feasible text
    candidate = latest_text_per_day[
        (latest_text_per_day['date'] >= input_start) &
        (latest_text_per_day['date'] < pred_start)
    ]

    if len(candidate) == 0:
        dropped += 1
        continue

    chosen = candidate.iloc[-1]

    sample = {
        "text": chosen['summary'],
        "text_timestamp": str(chosen['date_time']),

        # 👇 JSON string
        "input_timestamps": json.dumps([str(d) for d in input_dates]),
        "output_timestamps": json.dumps([str(d) for d in output_dates]),

        "input_window": json.dumps(build_channel_dict(input_data)),
        "output_window": json.dumps(build_channel_dict(output_data)),
    }

    samples.append(sample)

print(f"Kept: {len(samples)}, Dropped: {dropped}")

# =========================
# Split
# =========================
n = len(samples)
n_test = int(n * TEST_RATIO)

train_samples = samples[:n - n_test]
test_samples = samples[n - n_test:]

# =========================
# Save CSV
# =========================
os.makedirs(OUT_DIR, exist_ok=True)

pd.DataFrame(train_samples).to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
pd.DataFrame(test_samples).to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)

print("Done!")