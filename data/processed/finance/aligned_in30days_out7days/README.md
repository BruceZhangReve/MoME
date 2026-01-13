---
dataset_info:
  features:
  - name: input_timestamps
    sequence: float64
  - name: input_window
    sequence: float64
  - name: output_timestamps
    sequence: float64
  - name: output_window
    sequence: float64
  - name: text
    dtype: string
  - name: trend
    dtype: string
  - name: technical
    dtype: string
  - name: alignment
    dtype: string
  splits:
  - name: train
    num_bytes: 40760650
    num_examples: 525
  download_size: 22910094
  dataset_size: 40760650
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# MTBench: A Multimodal Time Series Benchmark


**MTBench** ([Huggingface](https://huggingface.co/collections/afeng/mtbench-682577471b93095c0613bbaa), [Github](https://github.com/Graph-and-Geometric-Learning/MTBench), [Arxiv](https://arxiv.org/pdf/2503.16858)) is a suite of multimodal datasets for evaluating large language models (LLMs) in temporal and cross-modal reasoning tasks across **finance** and **weather** domains.

Each benchmark instance aligns high-resolution time series (e.g., stock prices, weather data) with textual context (e.g., news articles, QA prompts), enabling research into temporally grounded and multimodal understanding.

## 🏦 Stock Time-Series and News Pair

This dataset contains aligned pairs of financial news articles and corresponding stock time-series data, designed to evaluate models on **event-driven financial reasoning** and **news-aware forecasting**.

### Pairing Process

Each pair is formed by matching a news article’s **publication timestamp** with a relevant stock’s **time-series window** surrounding the event

To assess the impact of the news, we compute the **average percentage price change** across input/output windows and label directional trends (e.g., `+2% ~ +4%`). A **semantic analysis** of the article is used to annotate the sentiment and topic, allowing us to compare narrative signals with actual market movement.

We observed that not all financial news accurately predicts future price direction. To quantify this, we annotate **alignment quality**, indicating whether the sentiment in the article **aligns with observed price trends**. Approximately **80% of the pairs** in the dataset show consistent alignment between news sentiment and trend direction.


###  Each pair includes:

- `"input_timestamps"` / `"output_timestamps"`: Aligned time ranges (5-minute resolution)
- `"input_window"` / `"output_window"`: Time-series data (OHLC, volume, VWAP, transactions)
- `"text"`: Article metadata
  - `content`, `timestamp_ms`, `published_utc`, `article_url`
  - Annotated `label_type`, `label_time`, `label_sentiment`
- `"trend"`: Ground truth price trend and bin labels
  - Percentage changes and directional bins (e.g., `"-2% ~ +2%"`)
- `"technical"`: Computed technical indicators
  - SMA, EMA, MACD, Bollinger Bands (for input, output, and overall windows)
- `"alignment"`: Label indicating semantic-trend consistency (e.g., `"consistent"`)



## 📦 Other MTBench Datasets

### 🔹 Finance Domain

- [`MTBench_finance_news`](https://huggingface.co/datasets/afeng/MTBench_finance_news)  
  20,000 articles with URL, timestamp, context, and labels

- [`MTBench_finance_stock`](https://huggingface.co/datasets/afeng/MTBench_finance_stock)  
  Time series of 2,993 stocks (2013–2023)

- [`MTBench_finance_aligned_pairs_short`](https://huggingface.co/datasets/afeng/MTBench_finance_aligned_pairs_short)  
  2,000 news–series pairs  
  - Input: 7 days @ 5-min  
  - Output: 1 day @ 5-min

- [`MTBench_finance_aligned_pairs_long`](https://huggingface.co/datasets/afeng/MTBench_finance_aligned_pairs_long)  
  2,000 news–series pairs  
  - Input: 30 days @ 1-hour  
  - Output: 7 days @ 1-hour

- [`MTBench_finance_QA_short`](https://huggingface.co/datasets/afeng/MTBench_finance_QA_short)  
  490 multiple-choice QA pairs  
  - Input: 7 days @ 5-min  
  - Output: 1 day @ 5-min

- [`MTBench_finance_QA_long`](https://huggingface.co/datasets/afeng/MTBench_finance_QA_long)  
  490 multiple-choice QA pairs  
  - Input: 30 days @ 1-hour  
  - Output: 7 days @ 1-hour

### 🔹 Weather Domain

- [`MTBench_weather_news`](https://huggingface.co/datasets/afeng/MTBench_weather_news)  
  Regional weather event descriptions

- [`MTBench_weather_temperature`](https://huggingface.co/datasets/afeng/MTBench_weather_temperature)  
  Meteorological time series from 50 U.S. stations

- [`MTBench_weather_aligned_pairs_short`](https://huggingface.co/datasets/afeng/MTBench_weather_aligned_pairs_short)  
  Short-range aligned weather text–series pairs

- [`MTBench_weather_aligned_pairs_long`](https://huggingface.co/datasets/afeng/MTBench_weather_aligned_pairs_long)  
  Long-range aligned weather text–series pairs

- [`MTBench_weather_QA_short`](https://huggingface.co/datasets/afeng/MTBench_weather_QA_short)  
  Short-horizon QA with aligned weather data

- [`MTBench_weather_QA_long`](https://huggingface.co/datasets/afeng/MTBench_weather_QA_long)  
  Long-horizon QA for temporal and contextual reasoning



## 🧠 Supported Tasks

MTBench supports a wide range of multimodal and temporal reasoning tasks, including:

- 📈 **News-aware time series forecasting**
- 📊 **Event-driven trend analysis**
- ❓ **Multimodal question answering (QA)**
- 🔄 **Text-to-series correlation analysis**
- 🧩 **Causal inference in financial and meteorological systems**



## 📄 Citation

If you use MTBench in your work, please cite:

```bibtex
@article{chen2025mtbench,
  title={MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering},
  author={Chen, Jialin and Feng, Aosong and Zhao, Ziyu and Garza, Juan and Nurbek, Gaukhar and Qin, Cheng and Maatouk, Ali and Tassiulas, Leandros and Gao, Yifeng and Ying, Rex},
  journal={arXiv preprint arXiv:2503.16858},
  year={2025}
}
