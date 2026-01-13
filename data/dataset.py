import json
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict, Counter
from pathlib import Path

import gc
import logging


class WeatherDataset(Dataset):

    def __init__(self, data_dir, tokenizer, input_seq_len=336, output_seq_len=72, max_text_length=2048):
        #Params:
        #    data_dir: The directory with Json files
        #    input_seq_len: Lenth of input time series (7/14)->[168/336]
        #    output_seq_len: Lenth of output time series (1/3)->[24,72]

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.max_text_length = max_text_length
        self.samples = []
        
        #validate the loaded DS
        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.parquet"))
        if not df_paths:
            raise ValueError(f"Parquet file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_parquet(path=df_path, engine="pyarrow")
            tqdm.write(f"Loaded: {df_path.name}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path.name} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_window', 'output_window', 'text', 'input_timestamps', 'output_timestamps', 'past_trend', 'future_trend']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_window'], dtype=torch.float32)
                output_window = torch.tensor(row['output_window'], dtype=torch.float32)

                input_timestamps = pd.to_datetime(row['input_timestamps']).astype('int64') / 1e9
                input_timestamps = np.array(input_timestamps, dtype=np.float32)
                output_timestamps = pd.to_datetime(row['output_timestamps']).astype('int64') / 1e9
                output_timestamps = np.array(output_timestamps, dtype=np.float32)

                text_data = row['text']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

                past_trend = row.get('past_trend', {})
                future_trend = row.get('future_trend', {})

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        
            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                "file_name": row['file_name'],
                "input_timestamps": input_timestamps,
                "output_timestamps": output_timestamps,
                "input_window": input_window,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
                "input_trend": past_trend,
                "output_trend": future_trend,
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_seq_len}, output_len: {self.output_seq_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]



class FinanceDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 tokenizer, 
                 input_len=312,   
                 output_len=78,
                 max_text_length=2048,
                 pad_value=0.0):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_value = pad_value
        self.input_len = input_len
        self.output_len = output_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.parquet"))
        if not df_paths:
            raise ValueError(f"Parquet file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_parquet(path=df_path, engine="pyarrow")
            tqdm.write(f"Loaded: {df_path.name}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path.name} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_window', 'output_window', 'text', 'input_timestamps', 'output_timestamps', 'trend', 'technical']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_window'], dtype=torch.float32)
                output_window = torch.tensor(row['output_window'], dtype=torch.float32)

                text_data = row['text']['content']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

                technical_data = row.get('technical', {})
                if not isinstance(technical_data, dict):
                    technical_data = {}  # safe loading of an errored case

                trend = row.get('trend', {})
                input_trend = trend['input_bin_label']
                output_trend = trend['output_bin_label']
                overall_trend = trend['overall_bin_label']

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        

            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                "file_name": row['file_name'],
                "input_timestamps": np.array(row['input_timestamps']),
                "output_timestamps": np.array(row['output_timestamps']),
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
                "technical": technical_data, 
                "input_trend": input_trend,
                "output_trend": output_trend,
                "overall_trend": overall_trend,
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_len}, output_len: {self.output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]



#########################################
##########For Data From TimeMMD##########
#########################################
class EnvironmentDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 tokenizer, 
                 input_len=7,   
                 output_len=1,
                 max_text_length=512,
                 pad_value=0.0):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_value = pad_value
        self.input_len = input_len
        self.output_len = output_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            #print(df.head(3))
            #print(type(df['input_seq'].iloc[0]))
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_seq', 'output_seq', 'text', 'input_start', 'input_end', 'output_start', 'output_end']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_seq'], dtype=torch.float32)
                output_window = torch.tensor(row['output_seq'], dtype=torch.float32)

                text_data = row['text']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        

            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                #"input_start": np.array(row['input_start']),
                #"input_end": np.array(row['input_end']),
                #"output_start": np.array(row['output_start']),
                #"output_end": np.array(row['output_end']),
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_len}, output_len: {self.output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]

def safe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    elif isinstance(x, list):
        return x
    else:
        return ast.literal_eval(str(x))


class EnergyDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 tokenizer, 
                 input_len=14,   
                 output_len=3,
                 max_text_length=512,
                 pad_value=0.0):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_value = pad_value
        self.input_len = input_len
        self.output_len = output_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_seq', 'output_seq', 'text']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_seq'], dtype=torch.float32)
                output_window = torch.tensor(row['output_seq'], dtype=torch.float32)

                text_data = row['text']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        

            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_len}, output_len: {self.output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]



class HealthUSDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 tokenizer, 
                 input_len=14,   
                 output_len=3,
                 max_text_length=512,
                 pad_value=0.0):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_value = pad_value
        self.input_len = input_len
        self.output_len = output_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_seq', 'output_seq', 'text']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_seq'], dtype=torch.float32)
                output_window = torch.tensor(row['output_seq'], dtype=torch.float32)

                text_data = row['text']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        

            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_len}, output_len: {self.output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]


class HealthAFRDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 tokenizer, 
                 input_len=14,   
                 output_len=3,
                 max_text_length=512,
                 pad_value=0.0):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_value = pad_value
        self.input_len = input_len
        self.output_len = output_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_seq', 'output_seq', 'text']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_seq'], dtype=torch.float32)
                output_window = torch.tensor(row['output_seq'], dtype=torch.float32)

                text_data = row['text']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        

            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_len}, output_len: {self.output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]


class SocialGoodDataset(Dataset):
    def __init__(self, 
                 data_dir, 
                 tokenizer, 
                 input_len=14,   
                 output_len=3,
                 max_text_length=512,
                 pad_value=0.0):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_value = pad_value
        self.input_len = input_len
        self.output_len = output_len
        self.samples = []

        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            df['input_seq'] = df['input_seq'].apply(safe_eval)
            df['output_seq'] = df['output_seq'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_seq', 'output_seq', 'text']
                for field in required_fields:
                    if field not in row:
                        raise KeyError(f"Sample {idx} Missing Key: {field}")
                    
                input_window = torch.tensor(row['input_seq'], dtype=torch.float32)
                output_window = torch.tensor(row['output_seq'], dtype=torch.float32)

                text_data = row['text']
                text_tokens = self.tokenizer(
                    text_data,
                    max_length=self.max_text_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = text_tokens["input_ids"].squeeze(0)
                attention_mask = text_tokens["attention_mask"].squeeze(0)

            except Exception as e:
                tqdm.write(f"Warning: Sample{idx}fails. Error: {str(e)}")
                continue
        

            input_ids = text_tokens["input_ids"].squeeze(0)
            attention_mask = text_tokens["attention_mask"].squeeze(0)

            samples.append({
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {self.input_len}, output_len: {self.output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]


def safe_eval(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    elif isinstance(x, list):
        return x
    else:
        return ast.literal_eval(str(x))
    
#########################################
##########For Data From TimeMMD##########
#########################################



# Let's do some testing, if it's 14->3, then the corresponding seq_le is 336->72
if __name__ == "__main__":
    #data_dir = "./processed/weather/aligned_in14days_out3days"
    #model_path = "../llm/Qwen1.5-MoE-A2.7B" #Qwen/Qwen1.5-MoE-A2.7B

    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #if tokenizer.pad_token is None:
    #    tokenizer.pad_token = tokenizer.eos_token
    
    #dataset = WeatherDataset(
    #    data_dir=data_dir,
    #    tokenizer=tokenizer,
    #    input_seq_len=336,
    #    output_seq_len=72
    #)
    
    #first_sample = dataset[0]
    #print(first_sample)

    #CUDA_VISIBLE_DEVICES=1,2 python dataset.py
    data_dir = "./processed/finance/aligned_in7days_out1days/data"
    tokenizer = AutoTokenizer.from_pretrained("../llm/Qwen1.5-MoE-A2.7B")  
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  
    dataset = FinanceDataset(data_dir=data_dir,
                             tokenizer=tokenizer,
                             input_len=390,
                             output_len=78,
                             max_text_length=2048)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_window'].shape}")
    print(f"Sample output shape: {sample['output_window'].shape}")
    print(f"Text input IDs shape: {sample['text_input_ids'].shape}")

