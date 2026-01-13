import os
import torch
import pickle
from torch.utils.data import DataLoader, random_split
from .dataset import WeatherDataset, FinanceDataset
#from dataset import WeatherDataset
from transformers import AutoTokenizer



def save_datasets(train_dataset, val_dataset, test_dataset, save_dir="./saved_datasets", suffix="weatherforecasting_long"):
    """
    Args:
        train_dataset/val_dataset/test_dataset: Dataset splitted by random Generator.
        save_dir: ...
        suffix: ...
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_paths = {
        "train": os.path.join(save_dir, f"train_dataset_{suffix}.pkl"),
        "val": os.path.join(save_dir, f"val_dataset_{suffix}.pkl"),
        "test": os.path.join(save_dir, f"test_dataset_{suffix}.pkl")
    }
    
    with open(save_paths["train"], "wb") as f:
        pickle.dump(train_dataset, f)
    with open(save_paths["val"], "wb") as f:
        pickle.dump(val_dataset, f)
    with open(save_paths["test"], "wb") as f:
        pickle.dump(test_dataset, f)
    
    print(f"\nThe splitted datasets are saved at {save_dir}.")
    for split, path in save_paths.items():
        print(f"- {split}: {path}")

def split_and_save_dataset(dataset, train_ratio=0.8, val_ratio=0.0, seed=7, save_dir="./saved_datasets", suffix="weatherforecasting_long"):
    """
    Construct dataloader for train/val/test
    Params:
        dataset: torch dataset
        batch_size: ...
        train_ratio: ...
        val_ratio: ...
        num_workers: ... 
        seed: ...
    Return:
        train_loader, val_loader, test_loader
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed) 
    )
    
    save_datasets(train_dataset, val_dataset, test_dataset, save_dir=save_dir, suffix=suffix)

    print(f"\nData Split: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}.")

    return train_dataset, val_dataset, test_dataset



def load_datasets(raw_dataset, train_ratio=0.8, val_ratio=0.0, seed=7, save_dir="./saved_datasets", suffix="weatherforecasting_long"):
    """
    Load the exsting dataset or create-save-load a dataset
    Args:
    ...
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    load_paths = {
        "train": os.path.join(save_dir, f"train_dataset_{suffix}.pkl"),
        "val": os.path.join(save_dir, f"val_dataset_{suffix}.pkl"),
        "test": os.path.join(save_dir, f"test_dataset_{suffix}.pkl")
    }

    all_files_exist = all(os.path.exists(path) for path in load_paths.values())

    new_max_text_length = raw_dataset.max_text_length
    
    if all_files_exist:
        print(f"\n Check existing data (if config changes)")
        with open(load_paths["train"], "rb") as f:
            train_dataset = pickle.load(f)

        if train_dataset[0]['text_input_ids'].shape[0] == new_max_text_length:
            print(f"\n Checked, load existing data")
            with open(load_paths["val"], "rb") as f:
                val_dataset = pickle.load(f)
            with open(load_paths["test"], "rb") as f:
                test_dataset = pickle.load(f)

        else:
            print(f"\n Changes! Create new data")
            train_dataset, val_dataset, test_dataset = split_and_save_dataset(
                dataset=raw_dataset,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
                save_dir=save_dir,
                suffix=suffix
            )

    else:
        print(f"\n No existing data, processing...")
        train_dataset, val_dataset, test_dataset = split_and_save_dataset(
            dataset=raw_dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            save_dir=save_dir,
            suffix=suffix
        )
    
    print(f"\nSuccessfully loaded datasets.")
    print(f"- Train: {len(train_dataset)} samples")
    print(f"- Val: {len(val_dataset)} samples")
    print(f"- Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset



def build_loader_from_saved(dataset, batch_size=4, num_workers=0, train_ratio=0.8, val_ratio=0.0, 
                            seed=7, save_dir="./saved_datasets", suffix="weatherforecasting_long",
                            train_shuffle=True):
    
    train_dataset, val_dataset, test_dataset = load_datasets(
        raw_dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        save_dir=save_dir,
        suffix=suffix
    )

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


# Let's do some testing, if it's 14->3, then the corresponding seq_le is 336->72
if __name__ == "__main__":
    data_dir = "./processed/weather/aligned_in14days_out3days"
    model_path = os.path.abspath("../llm/Qwen1.5-MoE-A2.7B")
    batch_size = 1
    train_ratio = 0.7
    val_ratio = 0 
    num_workers = 0 
    seed = 7 

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    raw_dataset = WeatherDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        input_seq_len=336,
        output_seq_len=72
    )

    # num_woeker better set to be 0, due to some warning
    train_loader, val_loader, test_loader = build_loader_from_saved(
        dataset=raw_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        save_dir="./saved_datasets/weather_forecasting",
        suffix="in14_out3" 
    )

    first_batch = next(iter(train_loader))
    print(first_batch['input_window'].shape) # [B,d], since it's single-variate
    print(first_batch['text_input_ids'].shape) # Every sample has a desciption
    print(first_batch['text_attention_mask'].sum(dim=1)) # check if `max_txt_len` is large enough
    print(first_batch['output_window'].shape)

