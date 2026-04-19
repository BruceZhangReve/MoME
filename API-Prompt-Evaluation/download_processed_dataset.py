from huggingface_hub import snapshot_download
import time

def download_dataset(repo_id, local_dir, max_workers=2):

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            max_workers=max_workers  
        )
        print(f"成功下载: {repo_id}\n")
    except Exception as e:
        print(f"下载失败 {repo_id}: {str(e)}\n")

    time.sleep(3)  


download_dataset(
    "afeng/MTBench_finance_aligned_pairs_short",
    "./data/processed/finance/aligned_in7days_out1days"
)
download_dataset(
    "afeng/MTBench_finance_aligned_pairs_long",
    "./data/processed/finance/aligned_in30days_out7days"
)
download_dataset(
    "afeng/MTBench_finance_QA_short",
    "./data/processed/finance/QAshort"
)
download_dataset(
    "afeng/MTBench_finance_QA_long",
    "./data/processed/finance/QAlong"
)


#download_dataset(
#    "afeng/MTBench_weather_aligned_pairs_short",
#    "./data/processed/weather/aligned_in7days_out1days"
#)
#download_dataset(
#    "afeng/MTBench_weather_aligned_pairs_long",
#    "./data/processed/weather/aligned_in14days_out3days"
#)
#download_dataset(
#    "afeng/MTBench_weather_QA_short",
#    "./data/processed/weather/QAshort"
#)
#download_dataset(
#    "afeng/MTBench_weather_QA_long",
#    "./data/processed/weather/QAlong"
#)

"""
from huggingface_hub import snapshot_download

# downloading processed finance dataset

## For Time Series Forecasting,Trend Prediction and Technical Indicator
snapshot_download("afeng/MTBench_finance_aligned_pairs_short", local_dir="./data/processed/finance/aligned_in7days_out1days", repo_type="dataset")
snapshot_download("afeng/MTBench_finance_aligned_pairs_long", local_dir="./data/processed/finance/aligned_in30days_out7days", repo_type="dataset")

## For MCQA and Correlation
snapshot_download("afeng/MTBench_finance_QA_short", local_dir="./data/processed/finance/QAshort", repo_type="dataset")
snapshot_download("afeng/MTBench_finance_QA_long", local_dir="./data/processed/finance/QAlong", repo_type="dataset")



# downloading processed weather dataset

## For Time Series Forecasting,Trend Prediction and Technical Indicator
snapshot_download("afeng/MTBench_weather_aligned_pairs_short", local_dir="./data/processed/weather/aligned_in7days_out1days", repo_type="dataset")
snapshot_download("afeng/MTBench_weather_aligned_pairs_long", local_dir="./data/processed/weather/aligned_in14days_out3days", repo_type="dataset")

## For MCQA
snapshot_download("afeng/MTBench_weather_QA_short", local_dir="./data/processed/weather/QAshort", repo_type="dataset")
snapshot_download("afeng/MTBench_weather_QA_long", local_dir="./data/processed/weather/QAlong", repo_type="dataset")
"""