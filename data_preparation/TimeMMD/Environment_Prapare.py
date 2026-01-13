# -*- coding: utf-8 -*-
# Generate English AQI news using DeepSeek Chat (Final Version, 2025.10)

import os, time, random, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, Any

# =====================
# Hyperparameters
# =====================
IN_SEQ   = 7
OUT_SEQ  = 1
STRIDE   = 8
TRAIN_PROP = 0.8
EMBARGO_WEEKS = max(IN_SEQ, OUT_SEQ)
NUM_PATH = "../../data/raw/TimeMMD/Environment/Environment.csv"

# =====================
# API Client
# =====================
DEEPSEEK_API_KEY = "..."
assert DEEPSEEK_API_KEY, "Please set: export DEEPSEEK_API_KEY='sk-xxxx'"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


# =====================
# Load numerical data
# =====================
df = pd.read_csv(NUM_PATH)
df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"]   = pd.to_datetime(df["end_date"])
df = df.sort_values("start_date").reset_index(drop=True)
assert "OT" in df.columns, "Environment.csv must contain 'OT' column."

# =====================
# Train/Test Split
# =====================
unique_dates = df["start_date"].drop_duplicates().sort_values().reset_index(drop=True)
split_idx = int(len(unique_dates) * TRAIN_PROP)
split_idx = max(IN_SEQ + OUT_SEQ, min(split_idx, len(unique_dates) - (IN_SEQ + OUT_SEQ)))

train_end_date = unique_dates.iloc[split_idx - 1]
test_begin_date = train_end_date + pd.Timedelta(weeks=EMBARGO_WEEKS)

train_df = df[df["start_date"] <= train_end_date].copy()
test_df  = df[df["start_date"] >  test_begin_date].copy()

def make_samples_stride(sub_df: pd.DataFrame,
                        in_len: int,
                        out_len: int,
                        stride: int,
                        use_col: str = "OT") -> pd.DataFrame:
    sub_df = sub_df.sort_values("start_date").reset_index(drop=True)
    items = []
    n = len(sub_df)
    for i in range(0, n - in_len - out_len + 1, stride):
        inp = sub_df.iloc[i:i+in_len]
        out = sub_df.iloc[i+in_len:i+in_len+out_len]
        items.append({
            "input_start":  inp["start_date"].iloc[0],
            "input_end":    inp["end_date"].iloc[-1],
            "output_start": out["start_date"].iloc[0],
            "output_end":   out["end_date"].iloc[-1],
            "input_seq":    inp[use_col].tolist(),
            "output_seq":   out[use_col].tolist(),
        })
    return pd.DataFrame(items)

train_samples = make_samples_stride(train_df, IN_SEQ, OUT_SEQ, STRIDE, use_col="OT")
test_samples  = make_samples_stride(test_df,  IN_SEQ, OUT_SEQ, STRIDE, use_col="OT")
print(f"[OK] train={len(train_samples)}, test={len(test_samples)} samples")


# =====================
# AQI Robust Profiling
# =====================
def _robust_sigma(x):
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    mean_abs = np.mean(np.abs(x)) + 1e-6
    return max(std, 0.5*iqr, 0.1*mean_abs, 1e-6)

def _level_from_ratio(r, low=0.02, high=0.06):
    if r < low: return "low"
    if r < high: return "medium"
    return "high"

def _variability_profile(x):
    x = np.asarray(x, dtype=float)
    mean_abs = np.mean(np.abs(x)) + 1e-6
    std_norm = np.std(x) / mean_abs
    rng_norm = (np.max(x) - np.min(x)) / mean_abs
    return {
        "variability": _level_from_ratio(std_norm),
        "amplitude": _level_from_ratio(rng_norm)
    }

def future_profile_from_out(input_seq, out_seq):
    x = np.asarray(input_seq, dtype=float)
    y = np.asarray(out_seq, dtype=float)
    sigma = _robust_sigma(x)

    # near-term
    z_next = (y[0] - x[-1]) / sigma
    if   z_next >= 1: dir_next = "worsening"
    elif z_next <= -1: dir_next = "improving"
    else: dir_next = "stable"
    sev_next = "pronounced" if abs(z_next)>=2 else "moderate" if abs(z_next)>=1 else "mild"

    # background (3-day mean diff)
    mean_first3, mean_last3 = np.mean(x[:3]), np.mean(x[-3:])
    z_bg = (mean_last3 - mean_first3) / sigma
    if   z_bg >= 1: dir_bg = "worsening"
    elif z_bg <= -1: dir_bg = "improving"
    else: dir_bg = "stable"
    str_bg = "pronounced" if abs(z_bg)>=2 else "moderate" if abs(z_bg)>=1 else "mild"

    vprof = _variability_profile(x)

    if dir_next == dir_bg and dir_next != "stable":
        pattern = f"persistent_{dir_next}"
    elif dir_next != "stable" and dir_bg != "stable" and dir_next != dir_bg:
        pattern = f"turning_to_{dir_next}"
    elif dir_next != "stable" and dir_bg == "stable":
        pattern = f"impulse_{dir_next}"
    elif dir_next == dir_bg == "stable" and (vprof["variability"]=="high" or vprof["amplitude"]=="high"):
        pattern = "stable_high_fluctuation"
    else:
        pattern = "mixed"

    return {
        "direction_next": dir_next,
        "severity_next": sev_next,
        "direction_bg": dir_bg,
        "strength_bg": str_bg,
        "variability": vprof["variability"],
        "amplitude": vprof["amplitude"],
        "pattern": pattern,
    }

def invert_direction(d):
    return {"improving": "worsening", "worsening": "improving"}.get(d, "stable")

# =====================
# Prompt & Generation
# =====================
def build_en_prompt(n_days, narrative_near, severity_near,
                    background, bg_strength, variability, amplitude, pattern):
    tone_map = {
        "improving": "(ventilation favorable; pollutant build-up limited)",
        "worsening": "(dispersion limited; accumulation more likely)",
        "stable": "(little change; overall steady circulation)"
    }
    pattern_map = {
        "persistent_improving": "Sustained dispersion improvements have characterized recent days.",
        "persistent_worsening": "Limited dispersion has persisted across recent days.",
        "turning_to_improving": "Recent observations indicate easing after a stagnant spell.",
        "turning_to_worsening": "Recent observations point to renewed stagnation after a comparatively better phase.",
        "impulse_improving": "A short-lived easing signal is visible without strong background support.",
        "impulse_worsening": "A brief stagnation signal appears absent a strong background tendency.",
        "stable_high_fluctuation": "Air quality has fluctuated widely day to day, though without a clear overall trend.",
        "mixed": "Signals are mixed across sites and times of day."
    }
    return f"""
You are an environmental journalist. Write ONE concise, professional English bulletin (150–200 words)
about RECENTLY OBSERVED air-quality conditions over the past {n_days} days.

Guidance (not a forecast):
- Near-term narrative: {narrative_near} {tone_map.get(narrative_near, "")}
- Severity: {severity_near}
- Background tendency: {background} ({bg_strength})
- Past-window variability: {variability}
- Amplitude: {amplitude}
- Pattern cue: {pattern_map.get(pattern, pattern_map['mixed'])}

Rules:
1) Describe observed drivers only: inversions, wind, humidity, transport, or anthropogenic sources.
2) No forecasts, numbers, or speculative language.
3) Keep factual, neutral, natural; one paragraph only.
Output only the paragraph.
""".strip()

BAN_PATTERNS_EN = [
    r"\bwill\b", r"\bexpect(ed|s)?\b", r"\bforecast(ed|s|ing)?\b", r"\blikely to\b",
    r"\bincrease(s|d|ing)?\b", r"\bdecrease(s|d|ing)?\b", r"\brising\b", r"\bfalling\b",
    r"\btrend(s|ing|ed)?\b", r"\btomorrow\b", r"\bnext\b", r"\bsoon\b", r"\bproject(ed|s|ing)?\b",
    r"\d+(\.\d+)?\s*%?"
]

def sanitize_en(text):
    s = text.strip()
    for pat in BAN_PATTERNS_EN:
        s = re.sub(pat, "[redacted]", s, flags=re.IGNORECASE)
    return s

def generate_one_en(prompt):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a careful environmental news writer. You only report observed context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7, max_tokens=300, stream=False
    )
    return sanitize_en(resp.choices[0].message.content.strip())

# =====================
# Batch Generate
# =====================
def attach_news_en_deepseek(df, p_consistent=0.8, seed=42):
    rng = random.Random(seed)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating AQI news"):
        prof = future_profile_from_out(row["input_seq"], row["output_seq"])
        is_consistent = 1 if rng.random() < p_consistent else 0
        narrative = prof["direction_next"] if is_consistent else invert_direction(prof["direction_next"])
        prompt = build_en_prompt(len(row["input_seq"]), narrative,
                                 prof["severity_next"], prof["direction_bg"], prof["strength_bg"],
                                 prof["variability"], prof["amplitude"], prof["pattern"])
        news = "(generation failed)"
        for _ in range(3):
            try:
                news = generate_one_en(prompt); break
            except Exception: time.sleep(1.2)
        new_row = row.copy()
        new_row["text"] = news; new_row["consistency"] = is_consistent
        rows.append(new_row)
    return pd.DataFrame(rows)

def preview_one_sample_news(df, idx=None, seed=123):
    rng = random.Random(seed)
    if idx is None: idx = rng.randrange(0, len(df))
    row = df.iloc[idx]; prof = future_profile_from_out(row["input_seq"], row["output_seq"])
    print(f"[Preview] row={idx} | near={prof['direction_next']}/{prof['severity_next']} | "
          f"bg={prof['direction_bg']}/{prof['strength_bg']} | var={prof['variability']} | "
          f"amp={prof['amplitude']} | pattern={prof['pattern']}")
    prompt = build_en_prompt(len(row["input_seq"]), prof["direction_next"],
                             prof["severity_next"], prof["direction_bg"], prof["strength_bg"],
                             prof["variability"], prof["amplitude"], prof["pattern"])
    return generate_one_en(prompt)

# =====================
# Run
# =====================
if __name__ == "__main__":
    print("[INFO] Generating training text ...")
    train_with_text = attach_news_en_deepseek(train_samples, p_consistent=0.9, seed=42)
    train_with_text.to_csv("../../data/processed/TimeMMD/Environment/Environment_train.csv", index=False)
    print("[INFO] Saved Environment_train.csv")

    print("[INFO] Generating test text ...")
    test_with_text = attach_news_en_deepseek(test_samples, p_consistent=0.9, seed=42)
    test_with_text.to_csv("../../data/processed/TimeMMD/Environment/Environment_test.csv", index=False)
    print("[INFO] Saved Environment_test.csv")

    # preview one
    #print(train_samples.iloc[1])
    #print(preview_one_sample_news(train_samples, idx=1))
