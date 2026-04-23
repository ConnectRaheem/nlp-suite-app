from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model once
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize(text: str, max_length: int = 150, min_length: int = 50) -> str:
    """
    Input  : raw text string
    Output : summarized text string
    """
    input_text = "summarize: " + text
    tokens = tokenizer.encode(input_text, return_tensors="pt",
                               max_length=512, truncation=True)
    summary_ids = model.generate(tokens,
                                  max_length=max_length,
                                  min_length=min_length,
                                  length_penalty=2.0,
                                  num_beams=4,
                                  early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)