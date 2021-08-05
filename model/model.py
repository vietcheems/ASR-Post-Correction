import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, PhobertTokenizer
import numpy as np
from datasets import load_dataset, load_metric
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


data_dir = '/home3/phungqv/post_correction/data/100k/word2num'
name = "word2num"
save_total_limit=7
num_train_epochs=17
batch_size = 16
logging_dir = os.path.join('/home3/phungqv/post_correction/model/logs', name)
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir)



# model_checkpoint = 'Helsinki-NLP/opus-mt-vi-en'
# model_checkpoint = '/home3/phungqv/data_generate/text-normalization/checkpoint-84561'

config = AutoConfig.from_pretrained('/home3/phungqv/post_correction/model/config.json')
model = AutoModelForSeq2SeqLM.from_config(config)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
tokenizer = PhobertTokenizer.from_pretrained('/home3/phungqv/post_correction/phobert_tokenizer')

train_path = os.path.join(data_dir, 'train.csv')
val_path = os.path.join(data_dir, 'val.csv')
dataset = load_dataset('csv', data_files={'train' : train_path, 'val' : val_path})
# dataset = load_dataset('csv', data_files={'train' : '/home3/phungqv/data_generate/data/finetune/train.csv', 
# 'val' : '/home3/phungqv/data_generate/data/finetune/val.csv'})
print('train first sample')
print(dataset['train'][0])
print('val first sample:')
print(dataset['val'][0])
# metric = load_metric("sacrebleu")

max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [str(ex) for ex in examples['input']]
    targets = [str(ex) for ex in examples['target']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    # print(model_inputs)
    model_inputs["labels"] = labels["input_ids"]
    # print(model_inputs)
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result = {"bleu": result["score"]}

    result = {}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

args = Seq2SeqTrainingArguments(
    name,
    evaluation_strategy = "epoch",
    # learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # weight_decay=0.01,
    save_total_limit=save_total_limit,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    fp16=True,
    logging_dir=logging_dir,
    logging_strategy='epoch',
    save_strategy='epoch'
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()