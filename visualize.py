from typing import List, Dict
import os

# Color palette for entity types
ENTITY_COLORS: Dict[str, str] = {
    "PER": "purple",
    "ORG": "blue",
    "LOC": "green",
    "MISC": "yellow",
}


def token_label_pairs_to_markdown(tokens: List[str], labels: List[str], title: str = None) -> str:
    """Convert token and BIO labels into an HTML-flavored Markdown string.

    - tokens: list of original tokens (already whitespace-separated)
    - labels: list of label strings like 'O', 'B-PER', 'I-PER'

    Returns a Markdown string containing colored spans for entities and a legend.
    """
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length")

    parts: List[str] = []
    cur_ent_tokens: List[str] = []
    cur_ent_type: str = None

    def close_entity():
        nonlocal cur_ent_tokens, cur_ent_type
        if not cur_ent_type:
            return
        text = " ".join(cur_ent_tokens).replace(" ", "%20")
        color = ENTITY_COLORS.get(cur_ent_type, "#ffdfe0")
        span = f'![](https://img.shields.io/badge/{text}-{color})'
        parts.append(span)
        cur_ent_tokens = []
        cur_ent_type = None

    for tok, lab in zip(tokens, labels):
        if lab == "O":
            # close any open entity
            if cur_ent_type:
                close_entity()
            parts.append(tok)
        elif lab.startswith("B-"):
            # begin new entity
            if cur_ent_type:
                close_entity()
            cur_ent_type = lab[2:]
            cur_ent_tokens = [tok]
        elif lab.startswith("I-"):
            ent = lab[2:]
            if cur_ent_type == ent and cur_ent_tokens:
                cur_ent_tokens.append(tok)
            else:
                # malformed I- tag (treat as B-)
                if cur_ent_type:
                    close_entity()
                cur_ent_type = ent
                cur_ent_tokens = [tok]
        else:
            # unknown tag, treat as plain
            if cur_ent_type:
                close_entity()
            parts.append(tok)

    if cur_ent_type:
        close_entity()

    # Build legend with only types that appear in the result
    legend_parts: List[str] = []
    for t in ENTITY_COLORS.keys():
        color = ENTITY_COLORS.get(t, "#ffdfe0")
        legend_parts.append(f'![](https://img.shields.io/badge/{t}-{color})')

    md_lines: List[str] = []
    if title:
        md_lines.append(f"### {title}")
    md_lines.append("")
    md_lines.append(" ".join(parts))
    md_lines.append("")
    if legend_parts:
        md_lines.append("**Legend:** " + " ".join(legend_parts))
    md_lines.append("")
    md_lines.append("---")

    return "\n".join(md_lines)


def save_markdown(md: str, path: str):
    md = md.replace("Ġ", "")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(md + "\n")
