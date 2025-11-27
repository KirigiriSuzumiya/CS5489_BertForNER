import torch
from datasets import load_dataset
from transformers import DataCollatorForTokenClassification

def tokenize_and_align_labels(examples, tokenizer):
    # Tokenize while keeping word-level mapping
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    # Prepare aligned token strings and labels (when provided)
    aligned_token_texts = []
    labels_aligned = []

    has_tags = "ner_tags" in examples
    ner_tags_list = examples.get("ner_tags", None)

    for i in range(len(tokenized_inputs["input_ids"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        aligned_tokens = []

        # Build aligned tokens and labels per token (same length as input_ids sequence)
        for j, word_idx in enumerate(word_ids):
            token_id = int(tokenized_inputs["input_ids"][i][j])
            # convert id to token string (keeps special tokens too)
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            aligned_tokens.append(token_str)

            if word_idx is None:
                # special token
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # first token of a word: use the corresponding word label if available
                if has_tags:
                    label_ids.append(ner_tags_list[i][word_idx])
                else:
                    label_ids.append(-100)
            else:
                # continuation sub-token: ignore for loss/metrics
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_token_texts.append(aligned_tokens)
        if has_tags:
            labels_aligned.append(label_ids)

    # Attach aligned tokens so callers (e.g., inference helpers) can use the exact token-level strings
    tokenized_inputs["aligned_tokens"] = aligned_token_texts
    if has_tags:
        tokenized_inputs["labels"] = labels_aligned

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