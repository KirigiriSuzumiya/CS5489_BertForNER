from transformers import AutoModelForTokenClassification, AutoTokenizer
from dataset import tokenize_and_align_labels
import torch
from visualize import token_label_pairs_to_markdown, save_markdown

class_labels = {}
for cls, idx in {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}.items():
    class_labels[idx] = cls

def test(sentance, model_name):
    if "roberta" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
        
    tokenized_input = tokenize_and_align_labels(
        {"tokens": [sentance.split()]},
        tokenizer,
    )
    # pull out aligned token strings (per-token strings aligned to model tokens)
    aligned_tokens = None
    if "aligned_tokens" in tokenized_input:
        # get first/example index
        aligned_tokens = tokenized_input.pop("aligned_tokens")[0]

    with torch.no_grad():
        input = {
            k: torch.tensor(v) for k, v in tokenized_input.items() if k in ["input_ids", "attention_mask"]
        }
        out = model(**input)
        out_classes = torch.argmax(out.logits, dim=-1)
    print("Input Sentance:", sentance.split())
    predicted = [class_labels[idx] for idx in out_classes.squeeze().tolist()[1:-1]]
    print("Predicted Classes:", predicted)

    # Generate Markdown visualization and save (appends)
    try:
        # Use aligned tokens (exclude special tokens) if available, else fall back to whitespace split
        if aligned_tokens:
            # aligned_tokens includes special tokens (e.g., cls/sep). We slice off first/last to match predicted
            vis_tokens = [t for t in aligned_tokens][1:-1]
        else:
            vis_tokens = sentance.split()

        md = token_label_pairs_to_markdown(vis_tokens, predicted)
        out_path = "outputs/ner_preview.md"
        save_markdown(md, out_path)
        print(f"Saved markdown preview to {out_path}")
    except Exception as e:
        print("Failed to generate markdown preview:", e)
    
        
        
if __name__ == "__main__":
    test("The City University of Hong Kong (CityUHK) is a public university in Kowloon Tong, Kowloon, Hong Kong.", "/Users/boyifan/code/CS5489-ML/project/outputs/roberta-large/checkpoint-17555")
    test("The 32nd APEC Leaders' Informal Meeting opened in Gyeongju, South Korea. At the venue, Hong Kong Chief Executive John Lee shook hands with South Korean President Lee Jae Myung, and the two exchanged greetings.", 
         "/Users/boyifan/code/CS5489-ML/project/outputs/roberta-large/checkpoint-17555")
    test(
        "Corporations from Hong Kong and the mainland have set up funds for relief efforts. Leading the way is the Hong Kong Jockey Club with a 100 million-dollar donation. Overnight -- the inferno engulfed homes as the flames tore through Wang Fuk Court, leaving smouldering ruins.Many displaced residents spent the night at the temporary shelter of the Church of Christ in China Fung Leung Kit Memorial Secondary School.",
        "/Users/boyifan/code/CS5489-ML/project/outputs/roberta-large/checkpoint-17555"
    )