# -*- coding: utf-8 -*-
# Public Health ILI (US) -> English News Synthesis (DeepSeek)
# v2: Hard-aligned prompt + post-generation alignment checks + retries
#
# Indicators used in prompt:
#   1) PAST_TREND (on input IN_SEQ)  -> direction/severity/slope_z
#   2) FLUCTUATION_INPUT (on input)  -> volatility_level/smoothness/vol_z/r2_linear/diff_z
#   3) NEAR_TERM_SHIFT               -> avg(out) vs avg(last k of input), direction/severity/delta_z
#   4) FUTURE_TREND (on output)      -> direction/severity/slope_z
#   5) RAW_INPUT_WINDOW (IN_SEQ points)
#
# Domain: CDC ILINet (weekly Influenza-Like Illness activity in the US).
# The journalist is instructed NOT to reveal numbers; outputs are sanitized; no forecasts.

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
p_consistent = 0.9       # proportion of consistent cases
EMBARGO_WEEKS = max(IN_SEQ, OUT_SEQ)

NUM_PATH = "../../data/raw/TimeMMD/HealthUS/HealthUS.csv"
OUT_DIR  = "../../data/processed/TimeMMD/HealthUS"
API_KEY  = "..."

# =====================
# API client (ENV)
# =====================
DEEPSEEK_API_KEY = API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
assert DEEPSEEK_API_KEY, "Please set API_KEY or export DEEPSEEK_API_KEY='sk-xxxx'"
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
    """
    Robust scale from input window only:
      sigma = max(std, 0.5*IQR, 1.4826*MAD, rel_floor_pct*|median|, 1e-6)
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    std = np.std(x)
    rel_floor = rel_floor_pct * (abs(med) + EPS)
    return max(std, 0.5 * iqr, 1.4826 * mad, rel_floor, 1e-6)

def _fit_line_slope(x: np.ndarray) -> float:
    """Least-squares slope vs time index (0..n-1)."""
    n = len(x)
    t = np.arange(n, dtype=float)
    denom = np.var(t) + EPS
    return np.cov(t, x, bias=True)[0, 1] / denom

def _net_change_z(slope: float, n: int, sigma: float) -> float:
    """
    Convert per-step slope to a scaled net-change z-score:
        net_change = slope * (n - 1); z = net_change / sigma.
    """
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
    """
    Detrend x with a line, then assess:
      - residual volatility (std of residuals) normalized by sigma -> vol_z
      - smoothness via R^2 of linear fit
      - roughness via mean absolute first diff normalized by sigma -> diff_z
    """
    t = np.arange(len(x), dtype=float)
    slope = _fit_line_slope(x)
    intercept = np.mean(x) - slope * np.mean(t)
    resid = x - (intercept + slope * t)

    sst = np.sum((x - np.mean(x))**2) + EPS
    sse = np.sum(resid**2)
    r2  = 1.0 - sse/sst

    vol_z = float(np.std(resid)) / (sigma + EPS)
    diff_z = float(np.mean(np.abs(np.diff(x)))) / (sigma + EPS)

    smooth = "smooth" if r2 >= 0.85 else "mixed" if r2 >= 0.60 else "choppy"
    vol_level = "low" if vol_z < 0.5 else "medium" if vol_z < 1.0 else "high"

    return {
        "volatility_level": vol_level,
        "smoothness": smooth,
        "vol_z": vol_z,
        "r2": float(r2),
        "diff_z": diff_z
    }

def invert_direction(d: str) -> str:
    return {"rising": "falling", "falling": "rising"}.get(d, "steady")

# =====================
# Metrics builder (ILI)
# =====================
def ili_metrics_from_io(input_seq, out_seq) -> Dict[str, Any]:
    """
    1) past-trend: slope_z on input (IN_SEQ)
    2) fluctuation: residual vol/smoothness/diff on input
    3) near-term shift: avg(out) vs avg(last len(out) of input), z using sigma(input)
    4) future-trend: slope_z on output (OUT_SEQ), z using sigma(input)
    """
    x = np.asarray(input_seq, dtype=float)
    y = np.asarray(out_seq, dtype=float)
    assert len(x) == IN_SEQ and len(y) == OUT_SEQ, "Unexpected window sizes."

    sigma_in = _robust_sigma(x)

    # 1) Past
    slope_in = _fit_line_slope(x)
    z_past = _net_change_z(slope_in, len(x), sigma_in)
    dir_past, sev_past = _classify_dir(z_past), _classify_sev(z_past)

    # 2) Fluctuation
    fluct = _fluctuation_metrics(x, sigma_in)

    # 3) Near-term shift
    k = len(y)
    z_near = (float(np.mean(y)) - float(np.mean(x[-k:]))) / (sigma_in + EPS)
    dir_near, sev_near = _classify_dir(z_near), _classify_sev(z_near)

    # 4) Future trend
    slope_out = _fit_line_slope(y)
    z_future = _net_change_z(slope_out, len(y), sigma_in)
    dir_future, sev_future = _classify_dir(z_future), _classify_sev(z_future)

    return {
        "past": {"direction": dir_past, "severity": sev_past, "slope_z": float(z_past)},
        "fluctuation": {
            "volatility_level": fluct["volatility_level"],
            "smoothness": fluct["smoothness"],
            "vol_z": float(fluct["vol_z"]),
            "r2_linear": float(fluct["r2"]),
            "diff_z": float(fluct["diff_z"]),
        },
        "near_term": {"direction": dir_near, "severity": sev_near, "delta_z": float(z_near)},
        "future": {"direction": dir_future, "severity": sev_future, "slope_z": float(z_future)},
    }

# =====================
# HARD-CONSTRAINED Prompt (ILI domain)
# =====================
def build_ili_prompt(input_seq, metrics, unit="week") -> str:

    n_periods = len(input_seq)

    locked_past   = metrics["past"]["direction"]       # rising/falling/steady
    locked_near   = metrics["near_term"]["direction"]  # rising/falling/steady
    locked_future = metrics["future"]["direction"]     # rising/falling/steady


    feature_block = {
        "PAST_TREND": {
            "direction": metrics["past"]["direction"],
            "severity":  metrics["past"]["severity"],
            "slope_z":   round(float(metrics["past"]["slope_z"]), 3),
        },
        "FLUCTUATION_INPUT": {
            "volatility_level": metrics["fluctuation"]["volatility_level"],
            "smoothness":       metrics["fluctuation"]["smoothness"],
            "vol_z":            round(float(metrics["fluctuation"]["vol_z"]), 3),
            "r2_linear":        round(float(metrics["fluctuation"]["r2_linear"]), 4),
            "diff_z":           round(float(metrics["fluctuation"]["diff_z"]), 3),
        },
        "NEAR_TERM_SHIFT": {
            "direction": metrics["near_term"]["direction"],
            "severity":  metrics["near_term"]["severity"],
            "delta_z":   round(float(metrics["near_term"]["delta_z"]), 3),
        },
        "FUTURE_TREND": {
            "direction": metrics["future"]["direction"],
            "severity":  metrics["future"]["severity"],
            "slope_z":   round(float(metrics["future"]["slope_z"]), 3),
        },
        "RAW_INPUT_WINDOW": [float(f"{v:.6g}") for v in input_seq],
    }

    rising_syn  = ["rising", "increasing", "moving higher", "upward", "elevated momentum"]
    falling_syn = ["falling", "declining", "moving lower", "downward", "easing back"]
    steady_syn  = ["steady", "broadly unchanged", "stable", "little changed", "flat"]

    return f"""
You are a public health reporter. Write ONE concise, professional English bulletin (120–160 words)
about OBSERVED U.S. influenza-like illness (ILI) conditions over the past {n_periods} {unit}s.

[LOCKED DIRECTIONS]
- PAST_TREND: {locked_past}
- NEAR_TERM_SHIFT: {locked_near}
- FUTURE_TREND: {locked_future}

[ALLOWED SYNONYMS]
- rising: {rising_syn}
- falling: {falling_syn}
- steady: {steady_syn}

[NON-NEGOTIABLE RULES]
1) Your narrative MUST align with ALL three LOCKED directions above. Do NOT contradict them.
2) If uncertain, prefer a 'steady' wording rather than inventing a trend.
3) Use only past or present-perfect tense; no forecasts or speculation.
4) Do NOT include any numbers (counts, percentages, rates) or dates.
5) Keep a neutral, factual tone; write ONE coherent paragraph.

[OBSERVED CONTEXT YOU MAY USE]
- outpatient ILI visit share and care-seeking patterns at reporting sites,
- virologic/lab signals and regional transmission differences already reported,
- co-circulation context (e.g., RSV/COVID) when noted,
- school schedules/holidays, weather shifts,
- vaccination campaigns and guidance already in effect,
- reporting cadence and known lags.

[FEATURES FOR INTERNAL REFERENCE ONLY — DO NOT QUOTE]
{feature_block}

Now write the paragraph. Do not show bullets or headings. Do not quote any of the [LOCKED ...] or [FEATURES] blocks.
""".strip()

# =====================
# Sanitizer + generator + Alignment check
# =====================
BAN_PATTERNS_HEALTH = [
    r"\bwill\b", r"\bexpect(ed|s)?\b", r"\bforecast(ed|s|ing)?\b", r"\blikely to\b",
    r"\bprojection(s)?\b", r"\boutlook\b", r"\btomorrow\b", r"\bnext\b", r"\bsoon\b",
    r"\banticipate(s|d|ing)?\b", r"\bcould\b", r"\bmay\b", r"\bmight\b",
    r"\d+(\.\d+)?\s*%?", r"\bpercent\b", r"\bper\s*cent\b",
    r"\bper\s*100k\b", r"\brate(s)?\b(?=\s*\d)", r"\bcount(s)?\b(?=\s*\d)",
    r"\bcase(s)?\b(?=\s*\d)", r"\bdeath(s)?\b(?=\s*\d)"
]
def sanitize_health_text(text: str) -> str:
    s = text.strip()
    for pat in BAN_PATTERNS_HEALTH:
        s = re.sub(pat, "[redacted]", s, flags=re.IGNORECASE)
    return s

# Direction lexicons for alignment checking
DIR_SYNONYMS = {
    "rising":  {"rising","increase","increasing","upward","moving higher","higher","elevated","climb","climbing","uptick","firming"},
    "falling": {"falling","decline","declining","downward","moving lower","lower","easing","ebb","drop","dropping","dip","softening"},
    "steady":  {"steady","stable","unchanged","flat","little changed","broadly unchanged","level","holding steady"}
}
def _mentions_direction(text: str, direction: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in DIR_SYNONYMS[direction])

def check_alignment(text: str, metrics: Dict[str, Any]) -> bool:
    ok_past   = _mentions_direction(text, metrics["past"]["direction"])
    ok_near   = _mentions_direction(text, metrics["near_term"]["direction"])
    ok_future = _mentions_direction(text, metrics["future"]["direction"])
    return bool(ok_past and ok_near and ok_future)

def generate_health_with_alignment(prompt: str, metrics: Dict[str, Any],
                                   model="deepseek-chat", temperature=0.2, max_tokens=260,
                                   max_retries: int = 3) -> str:

    base_system = "You are a careful public health reporter. You only describe observed context and avoid numbers."
    attempt = 0
    last_text = "(generation failed)"
    while attempt <= max_retries:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            last_text = sanitize_health_text(resp.choices[0].message.content.strip())
            if check_alignment(last_text, metrics):
                return last_text
            # 不对齐 => 加强提示再试
            corrective_hint = (
                f"\nYour last draft contradicted the LOCKED directions.\n"
                f"Rewrite ONE paragraph that EXPLICITLY reflects:\n"
                f"- PAST_TREND: {metrics['past']['direction']}\n"
                f"- NEAR_TERM_SHIFT: {metrics['near_term']['direction']}\n"
                f"- FUTURE_TREND: {metrics['future']['direction']}\n"
                f"Use only allowed synonyms and keep it observational without numbers."
            )
            prompt = prompt + corrective_hint
        except Exception:
            time.sleep(1.2)
        attempt += 1
    return last_text

# =====================
# Sampling & split utils
# =====================
def _build_samples(series: np.ndarray, in_len: int, out_len: int, stride: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(series)
    samples = []
    for i in range(0, n - (in_len + out_len) + 1, stride):
        x = series[i:i+in_len]
        y = series[i+in_len:i+in_len+out_len]
        samples.append((x, y))
    return samples

def _time_split(num_samples: int, train_prop: float, embargo: int) -> Tuple[np.ndarray, np.ndarray]:
    cut = int(np.floor(num_samples * train_prop))
    test_start = min(num_samples, cut + embargo)
    return np.arange(0, cut, dtype=int), np.arange(test_start, num_samples, dtype=int)

# =====================
# Column detection (ILI)
# =====================
DATE_CANDIDATES = ["week", "date", "start_date", "end_date"]

def _detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    date_col = None
    for c in df.columns:
        if c.lower() in DATE_CANDIDATES:
            date_col = c
            break
    # prefer 'OT' if present, else first numeric col not equal to date_col
    if "OT" in df.columns:
        val_col = "OT"
    else:
        numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
        assert numeric_cols, "No numeric ILI column found!"
        val_col = numeric_cols[0]
    return date_col, val_col

# =====================
# Batch generator
# =====================
def attach_ili_news(df: pd.DataFrame, p_consistent: float = 0.8, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating ILI news"):
        x, y = row["input_seq"], row["output_seq"]
        print(x, y)
        prof = ili_metrics_from_io(x, y)

        is_consistent = 1 if rng.random() < p_consistent else 0
        metrics = json.loads(json.dumps(prof))  # deep copy
        if not is_consistent:
            metrics["past"]["direction"]      = invert_direction(metrics["past"]["direction"])
            metrics["near_term"]["direction"] = invert_direction(metrics["near_term"]["direction"])
            metrics["future"]["direction"]    = invert_direction(metrics["future"]["direction"])
            # fluctuation texture unchanged

        prompt = build_ili_prompt(input_seq=x, metrics=metrics, unit="week")

        news = "(generation failed)"
        for _ in range(3):
            try:
                news = generate_health_with_alignment(prompt, metrics, temperature=0.2, max_tokens=260)
                break
            except Exception:
                time.sleep(1.2)
        print(news)

        out_rows.append({
            "input_seq": x, "output_seq": y,
            "text": news, "consistency": is_consistent,
            "metrics": prof
        })
    return pd.DataFrame(out_rows)

# =====================
# Preview helper
# =====================
def preview_one_sample(df: pd.DataFrame, idx: Optional[int] = None, seed: int = 123) -> str:
    rng = random.Random(seed)
    if idx is None:
        idx = rng.randrange(0, len(df))
    row = df.iloc[idx]
    prof = ili_metrics_from_io(row["input_seq"], row["output_seq"])
    print(
        f"[Preview] row={idx} | "
        f"past={prof['past']['direction']}/{prof['past']['severity']} (z={prof['past']['slope_z']:.3f}) | "
        f"near-term={prof['near_term']['direction']}/{prof['near_term']['severity']} (dz={prof['near_term']['delta_z']:.3f}) | "
        f"future={prof['future']['direction']}/{prof['future']['severity']} (z={prof['future']['slope_z']:.3f}) | "
        f"fluct={prof['fluctuation']['volatility_level']}/{prof['fluctuation']['smoothness']} "
        f"(vol_z={prof['fluctuation']['vol_z']:.3f}, r2={prof['fluctuation']['r2_linear']:.3f}, diff_z={prof['fluctuation']['diff_z']:.3f})"
    )
    prompt = build_ili_prompt(input_seq=row["input_seq"], metrics=prof, unit="week")
    return generate_health_with_alignment(prompt, prof, temperature=0.2, max_tokens=260)

# =====================
# Main
# =====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    assert os.path.exists(NUM_PATH), f"File not found: {NUM_PATH}"

    df_raw = pd.read_csv(NUM_PATH)
    date_col, val_col = _detect_columns(df_raw)
    if date_col:
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce")
        df_raw = df_raw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    series = df_raw[val_col].astype(float).to_numpy()
    assert len(series) >= (IN_SEQ + OUT_SEQ + 1), "Series too short for given windows."

    # sliding windows
    samples = _build_samples(series, IN_SEQ, OUT_SEQ, STRIDE)
    print(f"[Info] total samples={len(samples)} from {len(series)} points (IN={IN_SEQ}, OUT={OUT_SEQ}, STRIDE={STRIDE})")

    # time split with embargo
    train_idx, test_idx = _time_split(len(samples), TRAIN_PROP, EMBARGO_WEEKS)
    train_smp = [samples[i] for i in train_idx]
    test_smp  = [samples[i] for i in test_idx]
    print(f"[Split] Train={len(train_smp)}, Test={len(test_smp)}, Embargo={EMBARGO_WEEKS}")

    def smp_to_df(smp_list: List[Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        return pd.DataFrame([{"input_seq": x.tolist(), "output_seq": y.tolist()} for x, y in smp_list])

    train_df = smp_to_df(train_smp)
    test_df  = smp_to_df(test_smp)

    # generate texts
    train_with_text = attach_ili_news(train_df, p_consistent=p_consistent, seed=42)
    test_with_text  = attach_ili_news(test_df,  p_consistent=p_consistent, seed=42)

    # save
    train_out = os.path.join(OUT_DIR, "train_with_text.csv")
    test_out  = os.path.join(OUT_DIR, "test_with_text.csv")
    train_with_text.to_csv(train_out, index=False)
    test_with_text.to_csv(test_out, index=False)
    print(f"[Saved] {train_out} | {len(train_with_text)} rows")
    print(f"[Saved] {test_out} | {len(test_with_text)} rows")

    # preview one
    try:
        sample_text = preview_one_sample(train_df, idx=None)
        print("\n=== Preview Sample Text ===\n" + sample_text)
    except Exception as e:
        print(f"[Warn] Preview failed: {e}")

if __name__ == "__main__":
    main()