from pathlib import Path
import json
from datasets import load_dataset
from datasets import DatasetDict
import torch
from functools import partial


# opencompass datasets 
DATASET_NAMES = {
    "single_eq": "SingleEq",
    "addsub": "AddSub",
    "multiarith": "MultiArith",
    "gsm8k": "GSM8K",
    "aqua": "AQUA",
    "svamp": "SVAMP",
    "commonsense_qa": "Common",  # SenseQA
    "strategy_qa": "Strategy",  # QA
    "date_understanding": "Date",  # Understanding
    "tracking_shuffled_objects": "Shuffled",  # Objects
    "last_letter_concatenation": "Last Letter",  # (4 words)
    "coin_flip": "Coin Flip",  # (4 times)
}


def save_finetune_dataset(data_root_dir, shot, aug):
    
    """
    Generate dataset through merging relavant files

    :param shot: how many question are included in the file
    :pram aug: how many solutions are provided for one question
    """

    dataset_keys = list(DATASET_NAMES.keys())
    
    dataset_student = []
    dataset_vanilla = []
    
    for dataset_key in dataset_keys:
        # finetune_key_student = "F_zs_cot_t70_{}_{}shot_{}aug.jsonl".format(dataset_key, shot, aug)
        # #finetune_key_student = "F_zs_cot_t70_{}_{}shot.jsonl".format(dataset_key, shot)
        # finetune_key_vanilla = "F_{}_{}shot.jsonl".format(dataset_key, shot)
        
        finetune_key_student = "F_zs_cot_{}_{}shot".format(dataset_key, shot)
        finetune_key_vanilla = "F_{}_{}shot".format(dataset_key, shot)
                # Open the file for reading
        try:
            with open(data_root_dir / (finetune_key_student + ".jsonl"), 'r') as file:
                # Read each line in the file
                for line in file:
                    # Parse the JSON data and append it to the list
                    dataset_student.append(json.loads(line))
        except:
            print(data_root_dir / (finetune_key_student + "_1aug" + ".jsonl"))
            try:
                with open(data_root_dir / (finetune_key_student + "_1aug" + ".jsonl"), 'r') as file:
                    # Read each line in the file
                    for line in file:
                        # Parse the JSON data and append it to the list
                        dataset_student.append(json.loads(line))
            except:
                print(data_root_dir / (finetune_key_student + "_1aug" + ".jsonl"))
            
    
        with open(data_root_dir / (finetune_key_vanilla + ".jsonl"), 'r') as file:
            # Read each line in the file
            for line in file:
                # Parse the JSON data and append it to the list
                dataset_vanilla.append(json.loads(line))
            
    
    output_file_student = data_root_dir / 'student_{}shot_{}aug.json'.format(shot, aug)
    output_file_vanilla = data_root_dir / 'vanilla_{}shot.json'.format(shot)

    with open(output_file_student, 'w') as file:
        json.dump(dataset_student, file, indent=4)
        
    with open(output_file_vanilla, 'w') as file:
        json.dump(dataset_vanilla, file, indent=4)
    
    return output_file_student, output_file_vanilla



def load_train_val_dataset(filepath):
    
    """
    Load training and validation datasets from a JSON file.
    """

    filepath = str(filepath)
    ds = load_dataset("json", data_files= filepath)
    ds_train_devval = ds['train'].train_test_split(test_size=0.2, seed=42)
    # split the dataset
    train_dataset =  ds_train_devval['train']
    val_dataset = ds_train_devval['test']
    
    return train_dataset, val_dataset


def tokenize_dataset(example, tokenizer):

    """
    Tokenize and prepare datasets for training and validation
    """

    input_max_length = 512
    it = tokenizer(
        example["prompt"],
        max_length=input_max_length,
        truncation=True
    )
    
    iids = it["input_ids"]
    lids = tokenizer(
        example["completion"],
        max_length=512,
        truncation=True
    )["input_ids"]
    
    lengths = []
    input_ids = []
    attention_mask = []
    label_ids = []
    
    
    for iid, lid in zip(iids, lids):
        
        lengths.append(len(iid) + len(lid))
        input_ids.append(iid + lid)
        attention_mask.append([1] * (len(iid) + len(lid)))
        label_ids.append([-100] * len(iid) + lid)

    # Pad full sequences
    lengths = torch.tensor(lengths)
    pad_lengths = (lengths.max() - lengths).tolist()
    for i, l in enumerate(pad_lengths):
        # Apply left side padding
        # Why? https://github.com/huggingface/transformers/issues/3021#issuecomment-1231526631
        input_ids[i] = [tokenizer.pad_token_id] * l + input_ids[i]
        attention_mask[i] = [0] * l + attention_mask[i]
        label_ids[i] = [-100] * l + label_ids[i]
        
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(label_ids, dtype=torch.long),
    }


def prepare_dataset(data_dir, tokenizer):

    """
    Tokenize train and validation dataset
    """

    # load and tokenize dataset
    train_dataset, val_dataset = load_train_val_dataset(data_dir)
    
    tokenized_train_dataset = train_dataset.map(lambda examples: tokenize_dataset(examples, tokenizer), batched=True, batch_size=len(train_dataset), remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(lambda examples: tokenize_dataset(examples, tokenizer), batched=True, batch_size=len(val_dataset), remove_columns=train_dataset.column_names)
    
    return tokenized_train_dataset, tokenized_val_dataset
    



