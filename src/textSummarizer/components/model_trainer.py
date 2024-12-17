from transformers import TrainingArguments,Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from datasets import load_dataset,load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config=config
    def train(self):
        device="cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus=AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        dataset_samsum_pt=load_from_disk(self.config.data_path)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        trainer_args = TrainingArguments(
        output_dir=self.config.root_dir, 
        num_train_epochs=self.config.num_train_epochs, 
        warmup_steps=self.config.warmup_steps,
        per_device_train_batch_size=self.config.per_device_train_batch_size,  # Keep this small since you're training on CPU
        per_device_eval_batch_size=self.config.per_device_train_batch_size,  # Same for evaluation
        weight_decay=self.config.weight_decay, 
        logging_steps=self.config.logging_steps,
        evaluation_strategy=self.config.evaluation_strategy, 
        eval_steps=self.config.eval_steps, 
        save_steps=self.config.save_steps,
        gradient_accumulation_steps=self.config.gradient_accumulation_steps,  # Lowered gradient accumulation steps since CPU is slower
        use_cpu=True,  # This specifies you are using CPU for training (replaces `no_cuda=True`)
        fp16=False
        )
        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["test"],
                  eval_dataset=dataset_samsum_pt["validation"])
        trainer.train()
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
    

