<h1 align="center">
  <img src="asset/Git_Intuition.png" width="500" /></a><br>
  <b>Multi-Modal Time Series Prediction via Mixture of Modulated Experts</b><br>
</h1>


## Table of Contents

- [Table of Contents](#table-of-contents)
- [1. Abstract](#1-abstract)
- [2. Folder Structure](#2-folder-structure)
- [3. Dataset and Pre-Trained LLMs](#3-dataset-and-pre-trained-llms)
  - [Dependencies](#dependencies)
  - [Download Dataset](#download-dataset)
  - [Pre-trained LLMs](#pre-trained-llms)
- [4. Model Usage](#4-model-usage)
  - [Quick Run](#quick-run)
  - [Main Experiments](#main-experiments)
  - [Ablation Experiments](#ablation-experiments)
- [5. Experimental Results](#5-experimental-results)
  - [Results on MT-Bench and TimeMMD](#results-on-mt-bench-timemmd)
  - [Ablation Results](#ablation-results)
  - [Case Study](#case-study)
- [6. Contribution and Future Work](#6-contribution-and-future-work)
- [7. Additional Functions for Further Investigation](#7-additional-functions-for-further-investigation)
- [8. Acknowledgement](#8-acknowledgement)

## 1. Abstract

Real-world time series exhibit complex and evolving dynamics, making accurate forecasting extremely challenging. Recent multi-modal forecasting methods leverage textual information such as news reports to improve prediction, but most rely on token-level fusion that mixes temporal patches with language tokens in a shared embedding space. However, such fusion can be ill-suited when high-quality time–text pairs are scarce and when time series exhibit substantial variation in scale and characteristics, thus complicating cross-modal alignment. In parallel, Mixture-of-Experts (MoE) architectures have proven effective for both time series modeling and multi-modal learning, yet many existing MoE-based modality integration methods still depend on token-level fusion. To address this, we propose *Expert Modulation*, a new paradigm for multi-modal time series prediction that conditions both routing and expert computation on textual signals, enabling direct and efficient cross-modal control over expert behavior. Through comprehensive theoretical analysis and experiments, our proposed method demonstrates substantial improvements in multi-modal time series prediction.

## 2. Folder Structure

```
MoME/
|── API-Prompt-Evaluation           # Prompt-based method evaluation on MT-Bench
|── TS_Encoder                      # Different time series models (MoME, MMLinear, MiTransformer, etc.)
|── Uni-Model Study                 # Study on Uni-Modal MoE-based time series models
│── data/                           # Downloaded datasets
    ├── raw/                        # Text or timeseries only dataset
    ├── processed/                  # Task-specific dataset (Please download on webpage, or from supplementary materials)
│── saved_datasets/                 # The pre-partitioned datasets used in the experiments (may be deleted and creat new partitions)
│── data_preparation/               # Dataset preparation scripts
    ├── TimeMMD/                    # Scripts for TimeMMD data processing
    ├── finance/                    # Scripts for financial data processing
    ├── weather/                    # Scripts for weather data processing
│── llm/                            # Model and pre-trained parameters fot LLMs (also some utilily codes)
│── output/                         # Output directory (also contain some example parameters from author's experiments)
│── environment.yaml                # Dependencies
│── train_neue.py                   # The main training scripts for MoME
│── evaluate_neue.pu                # The main evaluation scripts for MoME
│── README.md                       # Project documentation
```

## 3. Dataset and Pre-Trained LLMs

Main experiments are conducted on two multi-modal time series benchmarks from MT-Bench, covering two domains: [weather, finance]; TimeMMD, convering four domains: [Environment, Energy, Infectious Disease, Social Good]. Each dataset consists of structured time series data and textual questions that require understanding of time-dependent trends.

### Dependencies

Run the following commands to create a conda environment for MTBench

```bash
git clone https://github.com/BruceZhangReve/MoME.git
cd MoME

conda env create -f environment.yml -n MoME
conda activate MoME
```

### Download Dataset

Dataset can be accquired at this page:
```
https://huggingface.co/datasets/lz245/MoME/tree/main
```
Under the "MoME" directory, there are "preprocessed" and "saved_datasets". Please download these files, and arrange them based on the "Folder Structure" described in Section 2.


For testing convinience, we also provide some *ready-to-use* data in the "./data/saved_datasets" directory (You may also delete these files, then when running the trainging script, it will automatically create these *.pkl* files in this directory as well).

### Pre-trained LLMs

In this codebase, GPT2 (for GPT4TS baseline) and Qwen-MoE[A2.7B] (for others) are utilized. Please use the following HuggingFace links to download them and put under the "./llm" directory.
```
https://huggingface.co/openai-community/gpt2
```
```
https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
```

## 4. Model Usage

<div align="center">
  <img src="asset/Git_Model.png" alt="Model Design" width="90%"/>
</div>
We summarize the proposed multi-modal learning architecture by figure above, also in comparison with conventional fusion methods.

Under *bfloat16* setting, the model training and evaluation can be performed on a single GPU that has 48GB memory (e.g., A6000). Under *float32*, it usually requires multiple GPUs to train the model. Here, we provide some example commands for training. There are a few key hyperparameters to mention. *--modulation* means that the *EiLM (Expert independent Linear Modulation)* is activated to enable multi-modal integration; *--n_experts* refers to the total number of experts for MoE; the *--topk* refers to the number of activated experts; *--instructor_query* refers to the number of instruct tokens (generated by LLMs) used to modulate the experts in the time series model. There are other hyperparameters such as *--finance_trend_choice* or *--weather_trend_choice*, please refer to the paper for them.

### Quick Run


### Main Experiments

#### Training
For instance, you can train a model that performs financial forecasting:
```
python train_neue.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --output_dir output/finance_forecast  --task finance_forecast --in_len 134 --out_len 33 --data_pkl_dir ./data/saved_datasets/finance_forecasting --dataset_path ./data/processed/finance/pair_in_30days_1hours_out_7days_1hours --data_suffix in30_out7 --epoch 8 --hidden_dim 32 --patch_len 8 --ts_encoder MoMe --use_bfloat16
```
You can train a model that performs financial trend prediction:
```
python train_neue.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --task finance_trend_prediction --in_len 134 --out_len 3 --finance_trend_choice 3way --data_pkl_dir ./data/saved_datasets/finance_trend_prediction --output_dir output/FT3 --dataset_path ./data/processed/finance/pair_in_30days_1hours_out_7days_1hours --data_suffix in30_out7 --epoch 10 --ts_encoder MoME --use_bfloat16 
```
You can train a model that performs weather forecasting:
```
python train_neue.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --task weather_forecast --in_len 168 --out_len 24 --data_pkl_dir ./data/saved_datasets/weather_forecasting --output_dir output/WF --dataset_path ./data/processed/weather/aligned_in7days_out1days --data_suffix in7_out1 --epoch 10 --hidden_dim 32 --patch_len 8 --ts_encoder MoMe --use_bfloat16
```
You can train a model that performs weather trend prediction:
```
python train_neue.py --instructor_query 2 --n_experts 4 --topk 2 --modulation --task weather_trend_prediction --in_len 168 --weather_trend_choice future --data_pkl_dir ./data/saved_datasets/weather_trend_prediction --output_dir output/WT --dataset_path ./data/processed/weather/aligned_in7days_out1days --data_suffix in7_out1 --epoch 5 --hidden_dim 32 --patch_len 8 --ts_encoder MoMe --use_bfloat16
```
You can train a model that performs other domain forecasting:
```
python train_neue.py --instructor_query 3 --n_experts 4 --topk 2 --modulation --task socialgood_forecast --in_len 14 --out_len 3 --output_dir output/SocialGood-MoME --dataset_path ./data/processed/TimeMMD/SocialGood/train  --epoch 8 --hidden_dim 32 --patch_len 4 --ts_encoder MoMe --use_bfloat16 
```

#### Evaluation
We provide some parameters trained in our experiments in this repo, and you can use the following command to directly make inference based on our parameters. However, you may also train your model and revise the command accordingly to test.

You can evaluate a model that performs financial forecasting (you can also add the command *--eval_mode random_sample --sample_seed 77*, which means infer on the specific sample 77, and the code will automatically draw a figure for you):
```
python evaluate_neue.py --task finance_forecast --in_len 134 --out_len 33 --data_pkl_dir ./data/saved_datasets/finance_forecasting --output_dir output/FF-L-MoME --dataset_path ./data/processed/finance/pair_in_30days_1hours_out_7days_1hours --data_suffix in30_out7  --checkpoint_path output/FF-L-MoME/ts_encoder_epoch7.pt --hidden_dim 32 --patch_len 8 --n_experts 4 --topk 2 --ts_encoder MoMe --use_bfloat16 --modulation --instructor_query 3
```
You can evaluate a model that performs financial trend prediction:
```
python evaluate_neue.py --task finance_trend_prediction --in_len 312 --finance_trend_choice 3way --data_pkl_dir ./data/saved_datasets/finance_trend_prediction --output_dir output/FT-S-MoME --dataset_path ./data/processed/finance/pair_in_7days_5minutes_out_1days_5minutes --data_suffix in7_out1 --checkpoint_path output/FT-S-MoME/ts_encoder_epoch9.pt --hidden_dim 32 --patch_len 8 --n_experts 4 --topk 2 --ts_encoder MoMe --use_bfloat16 --modulation --instructor_query 3
```

We will soon provide a more comprehensive instruction of more utilities of the codebase.

### Ablation Experiments
Coming Soon
#### Training
Coming Soon
#### Evaluation
Coming Soon

## 6. Experimental Results

We benchmark several state-of-the-art LLMs. The performance varies across different temporal reasoning tasks, highlighting areas for improvement in existing LLMs.

### Results on MT-Bench and TimeMMD

Evaluation on benchmark data. Note that for foundation models like Time-MoE, we only trians the prediction head while keeping other pre-trained parameters. The "DLinearP+LLM" means we use conventional fusion strategies to enabla multi-modal integration, and the time series encoder is "DLinearP".

|              | PatchTST |  Time-MoE*  | DLinearP+LLM | MoME (Ours) |
| ---------------------------- | :-------------------: | :-------------------: | :-------------------: | :-------------------: |
| **Stock Price Forecast (MAPE)**   |   3.832   | 4.564 |  4.010  |   **3.531**    |
| **Stock Price Trend (Acc)**   |   39.674   | 42.391 |  49.315  |    **66.849**   |
| **US Infectious Disease (MSE)**   |  1.503   | 0.789 |  0.587  |    **0.379**    |
| **Tempreture Forecast (MAE)** |  2.875 | 5.010 |  2.809  |   **2.620**   |


### Ablation Results
Ablation results, a fixed hyperparameter setting is applied to all experiments.

|              | MoME (w/o EiLM) |  MoME (Default)  | MoME (w/ RM) |
| ---------------------------- | :-------------------: | :-------------------: | :-------------------: | 
| **Stock Price Forecast (MAPE)**   |   3.758   | 3.531 |  **3.523**  | 
| **Stock Price Trend (Acc)**   |   45.108   | **66.849** |  61.413  |
| **US Infectious Disease (MSE)**   |  0.808   | **0.379** |  0.614  | 
| **Tempreture Forecast (MAE)** |  2.785 | 2.756 |  **2.741**  | 



### Case Study
We compare our method with some ablation models and baseline (keeping shared hyperparameters identical). One sample result is as follows:
<div align="center">
  <img src="asset/Git_Case.png" alt="Case Study" width="90%"/>
</div>


## 5. Contribution and Future Work

* Contribution 1: We propose *Expert Modulation*, a new paradigm for MMTSP that integrates temporal and textual signals by modulating expert routing and computation within an MoE framework, offering a principled alternative to token-level fusion for multi-modal learning
* Contribution 2: We develop a geometric interpretation of MoE and show that sparse routing can be understood as an energy-based truncation mechanism, providing theoretical insight into our modulation design.
* Contribution 3: We demonstrate the generality and effectiveness of our method across multiple time series backbones, achieving consistent improvements over representative baselines.
* Future Work: Generalizing MoME framework beyond two modalities (e.g., co-modulation), and beyond forecasting tasks.


## 7. Additional Functions for Further Investigation
We also provide some template scripts to generate time-series-aligned news report in the codebase, which allows users for further investigations. We will provide the instructions for using these codebases soon...


## 8. Acknowledgement
We also thank to the following repos for reference:
* MT-Bench: [https://github.com/Graph-and-Geometric-Learning/MTBench/tree/mainline](https://github.com/Graph-and-Geometric-Learning/MTBench/tree/mainline)
* TimeMMD: [https://github.com/AdityaLab/Time-MMD](https://www.google.com)
* Time-MoE: [https://github.com/Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE)
* Time-LLM: [https://github.com/KimMeen](https://github.com/KimMeen)
* One Fits All: [https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

If you find MoME useful, please consider citing our paper: NAN
