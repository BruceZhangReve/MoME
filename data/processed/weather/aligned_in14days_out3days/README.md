# MTBench: A Multimodal Time Series Benchmark


**MTBench** ([Huggingface](https://huggingface.co/collections/afeng/mtbench-682577471b93095c0613bbaa), [Github](https://github.com/Graph-and-Geometric-Learning/MTBench), [Arxiv](https://arxiv.org/pdf/2503.16858)) is a suite of multimodal datasets for evaluating large language models (LLMs) in temporal and cross-modal reasoning tasks across **finance** and **weather** domains.

Each benchmark instance aligns high-resolution time series (e.g., stock prices, weather data) with textual context (e.g., news articles, QA prompts), enabling research into temporally grounded and multimodal understanding.

## 📦 MTBench Datasets

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
