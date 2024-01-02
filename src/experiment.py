import argparse
import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, EarlyStoppingCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from utils import *
from constants import *




def eval_model(model, tokenizer, dataset,  eval_metrics):
    """
    Evaluates the model using the provided tokenizer and evaluation metrics on the dataset.

    Args:
    - model: The pre-trained model for evaluation.
    - tokenizer: The tokenizer associated with the model.
    - dataset: The dataset containing 'method' and 'method_name' features.
    - eval_metrics: The evaluation metrics object to update with predictions.

    Returns:
    - None
    """
    
    for i in range(len(dataset)):
                 method = dataset[i]['method']
                 method_name = dataset[i]['method_name'].replace('_<EOT>','').strip()
                 
                 input_ids = tokenizer(method, return_tensors="pt")["input_ids"].to("cuda")

                 output = model.generate(
                                input_ids = input_ids,
                                max_new_tokens=5,
                                attention_mask=torch.ones_like(input_ids),
                                pad_token_id=tokenizer.eos_token_id
                                              )[0]
                 prediction = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True).strip()
                 
                 eval_metrics.update(prediction, method_name)


            



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for LLM experiments")
    
    # Adding positional argument
    parser.add_argument("lang", type=str, help="language to fine tune model (current support for java, python)")
    args = parser.parse_args()
    language = args.lang

    
    
    print('train dataset creation')
    training_dataset = create_dataset(dataset_path + language +'_train.txt')

    print('eval dataset creation')
    eval_dataset = create_dataset(dataset_path + language + '_test.txt')


    # Model and tokenizer names
    base_model_name = "codellama/CodeLlama-7b-hf"


    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16

    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0}
    )

    

    print('Initial evaluation without model training')
    print('------------------------------------')
    eval_metrics = EvaluationMetrics()   
    eval_model(base_model, llama_tokenizer, eval_dataset,  eval_metrics)
    eval_metrics. display_metrics()
    print('------------------------------------')

    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    

    # LoRA Config
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Training Params
    train_params = TrainingArguments(
        output_dir ='./ckeckpoints/',
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=2500,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        evaluation_strategy="steps",
        eval_steps = 100,
        lr_scheduler_type="constant",
        load_best_model_at_end = True
    )


    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_dataset,
        eval_dataset = eval_dataset,
        peft_config=peft_parameters,
        formatting_func=formatting_prompts_func,
        tokenizer=llama_tokenizer,
        args=train_params
    )

    # Training
    fine_tuning.train()

    print('Evalaution after model fine-tuning')
    print('------------------------------------')
    eval_metrics = EvaluationMetrics()   
    eval_model(fine_tuning.model, llama_tokenizer, eval_dataset,  eval_metrics)
    eval_metrics. display_metrics()
    print('------------------------------------')



