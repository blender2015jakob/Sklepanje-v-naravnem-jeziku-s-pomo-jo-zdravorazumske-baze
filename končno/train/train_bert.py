import pandas as pd
from transformers import BertModel
import torch
from transformers import AutoTokenizer, BertConfig, TrainingArguments, DataCollatorWithPadding,\
	EvalPrediction, Trainer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
import datasets
import numpy as np
from typing import Dict
import time
from collections import Counter
import logging
import tensorflow as tf

#import dev, test, train (first row is column names)
location = "./Fixed_dataset/"

#import dev, test, train (first row is column names)
dev_data = datasets.load_dataset("csv", data_files=location + "dev.tsv", delimiter="\t")
dev_data = dev_data.rename_column("label", "labels")
train_data = datasets.load_dataset("csv", data_files=location + "train.tsv", delimiter="\t")
train_data = train_data.rename_column("label", "labels")

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {"entailment": 0, "neutral": 1, "contradiction": 2}

#convert labelst to ids
dev_data = dev_data.map(lambda example: {"labels": label2id[example["labels"]]})
train_data = train_data.map(lambda example: {"labels": label2id[example["labels"]]})

dev_data = dev_data["train"]
train_data = train_data["train"]

#TOKENIZER
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", batched=True)

#specil tokens
PREMISE_SPECIAL = "[PREMISE]"
HYPOTHESIS_SPECIAL = "[HYPOTHESIS]"

PREMISE_ADDITIONAL_DESCRIPTION = "[PREMISE_ADDITIONAL_DESCRIPTION]"
HYPOTHESIS_ADDITIONAL_DESCRIPTION = "[HYPOTHESIS_ADDITIONAL_DESCRIPTION]"

#add special tokens
num_added_toks = tokenizer.add_special_tokens({
	"additional_special_tokens": [PREMISE_SPECIAL, PREMISE_ADDITIONAL_DESCRIPTION, HYPOTHESIS_SPECIAL, HYPOTHESIS_ADDITIONAL_DESCRIPTION]
})

#PREPROCESS FUNCTION
def preprocess_function(examples):
	inputs = [f"{PREMISE_SPECIAL} {prem} {HYPOTHESIS_SPECIAL} {hyp}"
				for prem, hyp in zip(examples["premise"], examples["hypothesis"])]
	model_inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="tf")

	#labels in integers
	model_inputs["labels"] = tf.convert_to_tensor(examples["labels"], dtype=tf.int64)

	return model_inputs

#preprocess the data
tok_train = train_data.map(preprocess_function, batched=True)
tok_dev = dev_data.map(preprocess_function, batched=True)

#TRAINING
#gpu
device = "cuda" if torch.cuda.is_available() else "cpu"


config = BertConfig.from_pretrained("EMBEDDIA/sloberta",
									num_labels=3,
									id2label=id2label,
									label2id=label2id)

#get bert model
model_location = "EMBEDDIA/sloberta"
model = AutoModelForSequenceClassification.from_pretrained(model_location, config=config)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

#path of output of trained model
output_path = "sloberta_train"

training_args = TrainingArguments(
	output_dir=output_path,
	learning_rate=2e-5,
	num_train_epochs=100,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	load_best_model_at_end=True,
	evaluation_strategy="steps",
	save_strategy="steps",
	logging_steps=10,
	save_total_limit=3,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

""" def compute_metrics(eval_pred: EvalPrediction) -> Dict:
	predictions, labels = eval_pred
	accuracy = np.mean(predictions.argmax(axis=1) == labels)
	return {"accuracy": accuracy} """

#Print first example
print(train_data[0])

trainer = Trainer(
	model=model, args=training_args, tokenizer=tokenizer, data_collator=data_collator,
	#compute_metrics=compute_metrics,
	train_dataset=tok_train,
	eval_dataset=tok_dev
)

train_metrics = trainer.train()
t_end_train = time.time()

print(train_metrics)