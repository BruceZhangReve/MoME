import json
import ast
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm 
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from pathlib import Path


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import ast

    
class WeatherDataset(Dataset):

    def __init__(self, data_dir, tokenizer, max_text_length=2048):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.samples = []
        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            df['input_window'] = df['input_window'].apply(safe_eval)
            df['output_window'] = df['output_window'].apply(safe_eval)
            df['input_timestamps'] = df['input_timestamps'].apply(safe_eval)
            df['output_timestamps'] = df['output_timestamps'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")
        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_window', 'output_window', 'text', 'input_timestamps', 'output_timestamps', 'past_trend', 'future_trend']
                missing = [f for f in required_fields if f not in df.columns]
                if missing:
                    raise ValueError(f"CSV missing columns: {missing}")
                
                input_window = torch.tensor(row['input_window'], dtype=torch.float32)
                output_window = torch.tensor(row['output_window'], dtype=torch.float32)
                input_timestamps = pd.to_datetime(row['input_timestamps']).astype('int64') / 1e9
                input_timestamps = np.array(input_timestamps, dtype=np.float32)
                output_timestamps = pd.to_datetime(row['output_timestamps']).astype('int64') / 1e9
                output_timestamps = np.array(output_timestamps, dtype=np.float32)
                past_trend = row.get('past_trend', {})
                future_trend = row.get('future_trend', {})

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
        
        input_len = self.samples[0]["input_window"].shape[0]
        output_len = self.samples[0]["output_window"].shape[0]

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {input_len}, output_len: {output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]
    
    def safe_eval(self, x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        elif isinstance(x, list):
            return x
        else:
            return ast.literal_eval(str(x))


class FinanceDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_text_length=2048):  
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.samples = []
        self._load_data()

    def _load_data(self):
        df_paths = list(self.data_dir.glob("*.csv"))
        if not df_paths:
            raise ValueError(f"CSV file does not exists under {self.data_dir} directory")
        df_path = df_paths[0]  

        try:
            df = pd.read_csv(df_path)
            df['input_window'] = df['input_window'].apply(safe_eval)
            df['output_window'] = df['output_window'].apply(safe_eval)
            df['input_timestamps'] = df['input_timestamps'].apply(safe_eval)
            df['output_timestamps'] = df['output_timestamps'].apply(safe_eval)
            tqdm.write(f"Loaded: {df_path}, {len(df)} original samples in total")
        except Exception as e:
            raise ValueError(f"Load {df_path} fails: {str(e)}")

        
        samples = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                required_fields = ['input_window', 'output_window', 'text', 'input_timestamps', 
                                   'output_timestamps', 'input_trend', 'output_trend', 'overall_trend']
                missing = [f for f in required_fields if f not in df.columns]
                if missing:
                    raise ValueError(f"CSV missing columns: {missing}")         
                    
                input_window = torch.tensor(row['input_window'], dtype=torch.float32)
                output_window = torch.tensor(row['output_window'], dtype=torch.float32)
                input_timestamps = pd.to_datetime(row['input_timestamps']).astype('int64') / 1e9
                input_timestamps = np.array(input_timestamps, dtype=np.float32)
                output_timestamps = pd.to_datetime(row['output_timestamps']).astype('int64') / 1e9
                output_timestamps = np.array(output_timestamps, dtype=np.float32)
                input_trend = row.get('input_trend', {})
                output_trend = row.get('output_trend', {})
                overall_trend = row.get('overall_trend', {})

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

            samples.append({
                "file_name": row['file_name'],
                "input_timestamps": input_timestamps,
                "output_timestamps": output_timestamps,
                "input_window": input_window ,  
                "output_window": output_window,  
                "text_input_ids": input_ids,  
                "text_attention_mask": attention_mask,  
                "input_trend": input_trend,
                "output_trend": output_trend,
                "overall_trend": overall_trend,
            })

        self.samples = samples
        if not self.samples:
            raise ValueError("No valid samples loaded")

        input_len = self.samples[0]["input_window"].shape[0]
        output_len = self.samples[0]["output_window"].shape[0]

        tqdm.write(f"\nFinished dataset preparation")
        tqdm.write(f"Number of valid samples: {len(self.samples)}, input_len: {input_len}, output_len: {output_len}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range(total_num: {len(self.samples)})")
        return self.samples[idx]


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

