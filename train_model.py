from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from preprocessing_function import clean_text
from flask import jsonify
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np  
import nltk
from nltk.tokenize import sent_tokenize
import evaluate

metric = evaluate.load("accuracy")
# Load tokenizer dan model
model_id = "google/flan-t5-base"    
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

## TRAIN
class TrainItem(BaseModel):
    columns: List[str]
    parameters: Optional[dict] = {}
    data: List[Dict[str, str]]

def preprocess_dataframe(df):
    df = df.drop(df.index[:200]).reset_index(drop=True)

    # Buang duplikasi berdasarkan kolom 'Jawaban'
    df = df.drop_duplicates(subset=['Jawaban'], keep='first')
    
    # Buang baris yang mengandung nilai NaN di semua kolom
    df = df.dropna()
    
    # Buang kolom pertama (jika perlu)
    if df.shape[1] > 1:  # Pastikan ada lebih dari satu kolom
        df = df.drop(df.columns[0], axis=1)
        
    df['Skor'] = df['Skor'].astype(int).astype(str)
    
    return df

def train_model(data: TrainItem, columns: List[str] = [], parameters: dict = {}):
    data_records = data

    df_cleaned = preprocess_dataframe(data_records)

    df_train, df_test = train_test_split(pd.DataFrame.from_records(df_cleaned), test_size=0.3, random_state=42)
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    # Create DatasetDict with the desired format
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    # Menghapus kolom '__index_level_0__' dari "train" dataset
    dataset["train"] = dataset["train"].remove_columns('__index_level_0__')

    # Menghapus kolom '__index_level_0__' dari "test" dataset
    dataset["test"] = dataset["test"].remove_columns('__index_level_0__')

    # Setel path untuk menyimpan model dan tokenizer
    save_dir = f"./model/v{datetime.now().strftime('%d%m%Y')}"
    os.makedirs(save_dir, exist_ok=True)


    remove_col = ', '.join([f'{col.strip()}' for col in columns]).split(', ')
    filtered_columns = [col for col in columns if col != 'Skor']

    # Combine train and test datasets
    combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    # Tokenize the specified features column
    tokenized_inputs = combined_dataset.map(
        lambda x: tokenizer(*[x[col] for col in filtered_columns], truncation=True),
        batched=True,
        remove_columns=remove_col
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    # Tokenize the specified level column
    tokenized_targets = combined_dataset.map(
        lambda x: tokenizer(x['Skor'], truncation=True),
        batched=True,
        remove_columns=remove_col
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    
    # Apply preprocess_function to the dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(max_source_length, max_target_length+1, x, columns, tokenizer, padding="max_length"),
        batched=True,
        remove_columns=remove_col
    )

    #passing value from paramters dictionary
    if parameters:
        learning_rates = parameters["learning_rates"]
        num_train_epochs = parameters["num_train_epochs"]
        per_device_train_batch_size = parameters["per_device_train_batch_size"]
        per_device_eval_batch_size = 8
        save_total_limit = 2
    else:
        learning_rates = 3e-4
        num_train_epochs = 2
        per_device_train_batch_size = 8
        per_device_eval_batch_size = 8
        save_total_limit = 2

    # Argument pelatihan
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        learning_rate=learning_rates,
        num_train_epochs=num_train_epochs,
        logging_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )


    # Pelatihan model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )


    # Jalankan pelatihan
    print("START TRAIN")
    trainer.train()
    trainer.evaluate()

    # Simpan model dan tokenizer menggunakan pickle
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return {"status": "Model trained and saved successfully", "model_dir": save_dir}



def preprocess_function(sample, columns, padding="max_length"):
    """Tokenize the text and set the labels"""
    inputs = [f"{sample[col]}" for col in columns]
     # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, padding=padding, truncation=True)
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["Skor"], max_length=2, padding=padding, truncation=True)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Ensure predictions and labels are integers for accuracy calculation
    try:
        decoded_preds = [int(pred) for pred in decoded_preds]
        decoded_labels = [int(label) for label in decoded_labels]
    except ValueError as e:
        print(f"Error converting predictions/labels to int: {e}")
        return {"accuracy": 0}

    # Calculate metrics
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

def train_model_test(data: TrainItem, columns: List[str] = [], parameters: dict = {}):
    data_records = data

    df_train, df_test = train_test_split(pd.DataFrame.from_records(data_records), test_size=0.3, random_state=42)
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    # Create DatasetDict with the desired format
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    # Menghapus kolom '__index_level_0__' dari "train" dataset
    dataset["train"] = dataset["train"].remove_columns('__index_level_0__')

    # Menghapus kolom '__index_level_0__' dari "test" dataset
    dataset["test"] = dataset["test"].remove_columns('__index_level_0__')

    # Setel path untuk menyimpan model dan tokenizer
    save_dir = f"./model/v{datetime.now().strftime('%d%m%Y')}"
    os.makedirs(save_dir, exist_ok=True)

    # Load tokenizer dan model
    model_id = "google/flan-t5-base"    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)


    remove_col = ', '.join([f'{col.strip()}' for col in columns]).split(', ')
    filtered_columns = [col for col in columns if col != 'Level']

    # Combine train and test datasets
    combined_dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    # Tokenize the specified features column
    tokenized_inputs = combined_dataset.map(
        lambda x: tokenizer(*[x[col] for col in filtered_columns], truncation=True),
        batched=True,
        remove_columns=remove_col
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

    # Tokenize the specified level column
    tokenized_targets = combined_dataset.map(
        lambda x: tokenizer(x['Level'], truncation=True),
        batched=True,
        remove_columns=remove_col
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    
    # Apply preprocess_function to the dataset
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(max_source_length, max_target_length+1, x, columns, tokenizer, padding="max_length"),
        batched=True,
        remove_columns=remove_col
    )

    #passing value from paramters dictionary
    if parameters:
        num_train_epochs = parameters["num_train_epochs"]
        per_device_train_batch_size = parameters["per_device_train_batch_size"]
        per_device_eval_batch_size = parameters["per_device_eval_batch_size"]
        save_total_limit = parameters["save_total_limit"]
    else:
        num_train_epochs = 2
        per_device_train_batch_size = 8
        per_device_eval_batch_size = 8
        save_total_limit = 2

    # Argument pelatihan
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        predict_with_generate=True,
        learning_rate=3e-4,
        num_train_epochs=num_train_epochs,
        logging_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        push_to_hub=False,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )


    # Pelatihan model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )


    # Jalankan pelatihan
    print("START TRAIN")
    trainer.train()

    # Simpan model dan tokenizer menggunakan pickle
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return {"status": "Model trained and saved successfully", "model_dir": save_dir}