from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

from dataset import get_conll2003_dataset
from utils import Evaluator

def train(model_name):
    if "roberta" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds, data_collator, label_list = get_conll2003_dataset(tokenizer, model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    evaluator = Evaluator(label_list=label_list)

    training_args = TrainingArguments(
        output_dir=f"outputs/{model_name.split('/')[-1]}",
        learning_rate=2e-5,
        per_device_train_batch_size=4 if "large" in model_name else 16,
        per_device_eval_batch_size=4 if "large" in model_name else 16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=10,
        logging_strategy="steps",
        report_to=["tensorboard"],
        
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=evaluator,
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="test")
    return eval_results