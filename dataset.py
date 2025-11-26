import torch
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_conll2003_dataset(tokenizer, model_name):
    tokenizer = tokenizer
    ds = load_dataset("BramVanroy/conll2003")
    print("example data:", ds["train"][0])
    
    label_list = ds["train"].features[f"ner_tags"].feature.names
    print("label list:", label_list)
    
    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer":tokenizer})
    if "roberta" in model_name.lower():
        tokenized_ds = tokenized_ds.select_columns(["input_ids", "attention_mask", "labels"])
    else:
        tokenized_ds = tokenized_ds.select_columns(["input_ids", "token_type_ids", "attention_mask", "labels"])
    tokenized_ds = tokenized_ds.with_format(type="torch")
    # dataloader = torch.utils.data.DataLoader(ds, batch_size=32)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return tokenized_ds, data_collator, label_list