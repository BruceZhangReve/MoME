# -*- coding: utf-8 -*-
# Weekly Gasoline Price -> English News Synthesis (DeepSeek)
# v5: adds OUT_DIR hyperparameter
#
# Generates time-window samples from Energy.csv, computes four metrics:
#   1) PAST_TREND (input 14 weeks)
#   2) FLUCTUATION_INPUT (smoothness, volatility)
#   3) NEAR_TERM_SHIFT (avg(out) vs avg(last 3 of input))
#   4) FUTURE_TREND (output 3 weeks slope)
# Produces English bulletins for each sample using DeepSeek API.

import os, time, random, re, json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =====================
# Hyperparameters
# =====================
IN_SEQ   = 14
OUT_SEQ  = 3
STRIDE   = 2          
TRAIN_PROP = 0.8
p_consistent = 0.9    # proportion of consistent cases
EMBARGO_WEEKS = max(IN_SEQ, OUT_SEQ)

NUM_PATH = "../../data/raw/TimeMMD/Energy/Energy.csv"
OUT_DIR  = "../../data/processed/TimeMMD/Energy"
API_KEY  = "..."

# =====================
# API client (ENV)
# =====================
DEEPSEEK_API_KEY = API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
assert DEEPSEEK_API_KEY, "Please set API_KEY here or export DEEPSEEK_API_KEY='sk-xxxx'"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# =====================
# Robust stats & helpers
# =====================
EPS = 1e-12

def _level_from_ratio(r: float, low: float = 0.02, high: float = 0.06) -> str:
    if r < low: return "low"
    if r < high: return "medium"
    return "high"

def _robust_sigma(x: np.ndarray, rel_floor_pct: float = 0.005) -> float:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    std = np.std(x)
    rel_floor = rel_floor_pct * (abs(med) + EPS)
    return max(std, 0.5 * iqr, 1.4826 * mad, rel_floor, 1e-6)

def _fit_line_slope(x: np.ndarray) -> float:
    n = len(x)
    t = np.arange(n, dtype=float)
    denom = np.var(t) + EPS
    return np.cov(t, x, bias=True)[0, 1] / denom

def _net_change_z(slope: float, n: int, sigma: float) -> float:
    return (slope * max(n - 1, 1)) / (sigma + EPS)

def _classify_dir(z: float) -> str:
    if z >= 1.0: return "rising"
    if z <= -1.0: return "falling"
    return "steady"

def _classify_sev(z: float) -> str:
    az = abs(z)
    if az >= 2.0: return "pronounced"
    if az >= 1.0: return "moderate"
    return "mild"

def _fluctuation_metrics(x: np.ndarray, sigma: float) -> Dict[str, Any]:
    t = np.arange(len(x), dtype=float)
    slope = _fit_line_slope(x)
    intercept = np.mean(x) - slope * np.mean(t)
    resid = x - (intercept + slope * t)
    sst = np.sum((x - np.mean(x))**2) + EPS
    sse = np.sum(resid**2)
    r2  = 1.0 - sse / sst
    vol_z = np.std(resid) / (sigma + EPS)
    diff_z = np.mean(np.abs(np.diff(x))) / (sigma + EPS)
    smooth = "smooth" if r2 >= 0.85 else "mixed" if r2 >= 0.6 else "choppy"
    vol_level = "low" if vol_z < 0.5 else "medium" if vol_z < 1.0 else "high"
    return {"volatility_level": vol_level, "smoothness": smooth, "vol_z": vol_z, "r2": r2, "diff_z": diff_z}

def invert_price_direction(d: str) -> str:
    return {"rising": "falling", "falling": "rising"}.get(d, "steady")

# =====================
# Metrics builder
# =====================
def price_profile_from_out(x, y) -> Dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma_in = _robust_sigma(x)

    # Past-trend
    slope_in = _fit_line_slope(x)
    z_past = _net_change_z(slope_in, len(x), sigma_in)
    dir_past, sev_past = _classify_dir(z_past), _classify_sev(z_past)

    # Fluctuation
    fluct = _fluctuation_metrics(x, sigma_in)

    # Near-term shift
    k = len(y)
    z_near = (np.mean(y) - np.mean(x[-k:])) / (sigma_in + EPS)
    dir_near, sev_near = _classify_dir(z_near), _classify_sev(z_near)

    # Future-trend
    slope_out = _fit_line_slope(y)
    z_future = _net_change_z(slope_out, len(y), sigma_in)
    dir_future, sev_future = _classify_dir(z_future), _classify_sev(z_future)

    return {
        "past": {"direction": dir_past, "severity": sev_past, "slope_z": float(z_past)},
        "fluctuation": {**fluct},
        "near_term": {"direction": dir_near, "severity": sev_near, "delta_z": float(z_near)},
        "future": {"direction": dir_future, "severity": sev_future, "slope_z": float(z_future)},
    }

# =====================
# Prompt builder
# =====================
def build_price_prompt(input_seq, metrics, unit="week") -> str:
    n_periods = len(input_seq)
    def r2(x): return round(float(x), 4)
    def rz(x): return round(float(x), 3)
    feature_block = {
        "PAST_TREND": metrics["past"],
        "FLUCTUATION_INPUT": metrics["fluctuation"],
        "NEAR_TERM_SHIFT": metrics["near_term"],
        "FUTURE_TREND": metrics["future"],
        "RAW_INPUT_WINDOW": [float(f"{v:.6g}") for v in input_seq],
    }
    return f"""
You are a business journalist. Write ONE concise, professional English bulletin (120–180 words)
about OBSERVED gasoline retail price conditions over the past {n_periods} {unit}s.
Indicators (categorical and numeric features are provided for context; DO NOT reveal numbers):
{feature_block}
Rules:
- Describe observed market drivers only (crude benchmarks, refinery ops, inventories, demand, weather).
- Use past/present-perfect tense only; avoid forecasts.
- Do NOT include numbers, percentages, or currency.
- Write one neutral, factual paragraph.
Output only the paragraph.
""".strip()

# =====================
# Sanitizer + generator
# =====================
BAN_PATTERNS_PRICE = [
    r"\bwill\b", r"\bexpect(ed|s)?\b", r"\bforecast(ed|s|ing)?\b", r"\blikely to\b",
    r"\bprojection(s)?\b", r"\boutlook\b", r"\btomorrow\b", r"\bnext\b", r"\bsoon\b",
    r"\banticipate(s|d|ing)?\b", r"\bcould\b", r"\bmay\b", r"\bmight\b",
    r"\$\s*\d+(\.\d+)?", r"\d+(\.\d+)?\s*%?", r"\bbarrel(s)?\b", r"\bMMbbl(s)?\b"
]
def sanitize_price_text(text: str) -> str:
    for pat in BAN_PATTERNS_PRICE:
        text = re.sub(pat, "[redacted]", text, flags=re.IGNORECASE)
    return text.strip()

def generate_one_price(prompt: str, model="deepseek-chat", temperature=0.7, max_tokens=280) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful markets reporter; avoid numbers and forecasts."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return sanitize_price_text(resp.choices[0].message.content)

# =====================
# Dataset builder
# =====================
def _build_samples(series: np.ndarray, in_len: int, out_len: int, stride: int):
    n = len(series)
    samples = []
    for i in range(0, n - (in_len + out_len) + 1, stride):
        samples.append((series[i:i+in_len], series[i+in_len:i+in_len+out_len]))
    return samples

def _time_split(num_samples: int, train_prop: float, embargo: int):
    cut = int(np.floor(num_samples * train_prop))
    test_start = min(num_samples, cut + embargo)
    return np.arange(0, cut), np.arange(test_start, num_samples)

# =====================
# Main
# =====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(NUM_PATH)
    # pick numeric column
    val_col = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][0]
    series = df[val_col].astype(float).to_numpy()
    samples = _build_samples(series, IN_SEQ, OUT_SEQ, STRIDE)
    print(f"[Info] {len(samples)} total samples built from {len(series)} points.")

    # split train/test
    train_idx, test_idx = _time_split(len(samples), TRAIN_PROP, EMBARGO_WEEKS)
    train_smp = [samples[i] for i in train_idx]
    test_smp  = [samples[i] for i in test_idx]
    print(f"[Split] Train={len(train_smp)}, Test={len(test_smp)}, Embargo={EMBARGO_WEEKS}")

    def run_batch(smp_list, seed, tag):
        rng = random.Random(seed)
        rows = []
        for x, y in tqdm(smp_list, desc=f"Generating {tag}"):
            prof = price_profile_from_out(x, y)
            metrics = json.loads(json.dumps(prof))
            if rng.random() > p_consistent:
                for k in ("past", "near_term", "future"):
                    metrics[k]["direction"] = invert_price_direction(metrics[k]["direction"])
            prompt = build_price_prompt(x, metrics)
            try:
                text = generate_one_price(prompt)
            except Exception:
                time.sleep(1.5)
                text = "(generation failed)"
            rows.append({"input_seq": x.tolist(), "output_seq": y.tolist(), "text": text, "metrics": prof})
        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"{tag}_with_text.csv")
        out_df.to_csv(out_path, index=False)
        print(f"[Saved] {out_path} ({len(out_df)} samples)")

    run_batch(train_smp, 42, "train")
    run_batch(test_smp, 42, "test")

if __name__ == "__main__":
    main()
