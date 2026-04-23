import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> dict:
    """
    Input  : raw text string
    Output : dict {label: [entity1, entity2, ...]}
    """
    doc = nlp(text)
    result = {}
    for ent in doc.ents:
        result.setdefault(ent.label_, []).append(ent.text)
    return result

def get_entity_dataframe(text: str):
    import pandas as pd
    doc = nlp(text)
    return pd.DataFrame([
        {
            "entity"     : ent.text,
            "label"      : ent.label_,
            "description": spacy.explain(ent.label_)
        }
        for ent in doc.ents
    ])