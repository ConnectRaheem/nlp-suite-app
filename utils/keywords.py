from keybert import KeyBERT

kw_model = KeyBERT(model='all-MiniLM-L6-v2')

def extract_keywords(text: str, top_n: int = 5) -> list:
    """
    Input  : raw text string, number of keywords
    Output : list of tuples [(keyword, score), ...]
    """
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n,
        use_mmr=True,
        diversity=0.5
    )
    return keywords