python run.py --datasets gsm8k_gen \
--model-kwargs device_map='auto' \
--peft-path "/home/featurize/work/AIPI590-Project1/outputs/gpt2-finetune-2024-03-21-08-13/checkpoint-1000" \
--hf-path "openai-community/gpt2" \
--max-seq-len 1024 \
--max-out-len 100 \
--batch-size 128  \
--max-partition-size 40000 \
--max-num-workers 32 \
--num-gpus 1