from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from data_utils import load_train_val_dataset, tokenize_dataset
from data_utils import save_finetune_dataset
import wandb, os
from train import finetune
import argparse
from pathlib import Path

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_student", type=str, default=True)
    parser.add_argument("--shot", type=int, default=128)
    parser.add_argument("--aug", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--logging", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--data_root_dir", type=str, default="./data")
    args = parser.parse_args()

    
    # create dataset
    data_dir_student, data_dir_vanilla = save_finetune_dataset(Path(args.data_root_dir), args.shot, args.aug)

    if args.finetune_student == True:
        data_dir = data_dir_student
    else:
        data_dir = data_dir_vanilla
    

    
    # model quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # load model and tokenizer 
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", quantization_config=bnb_config, device_map={'':torch.cuda.current_device()})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
        
    
    finetune(model, tokenizer, Path(data_dir), args)
    

    
if __name__ == "__main__":
    
    # login wandb
    wandb.login()
    wandb_project = "gpt2-finetune"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    main()
    
    
    