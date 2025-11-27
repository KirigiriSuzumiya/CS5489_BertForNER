from train import train

if __name__ == "__main__":
    metrics = {}
    for model_name in [
        "google-bert/bert-base-uncased",
        # "google-bert/bert-large-uncased",
        # "FacebookAI/roberta-base",
        # "FacebookAI/roberta-large",
    ]:
        print(f"Training model: {model_name}")
        curr_metrics = train(model_name)
        metrics[model_name] = curr_metrics
        print(f"Metrics for {model_name}: {curr_metrics}")
    print("All metrics:", metrics)