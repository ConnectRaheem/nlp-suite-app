import textstat

def get_text_stats(text: str) -> dict:
    """
    Input  : raw text string
    Output : dict of all readability + complexity scores
    """
    return {
        "word_count"          : textstat.lexicon_count(text, removepunct=True),
        "sentence_count"      : textstat.sentence_count(text),
        "syllable_count"      : textstat.syllable_count(text),
        "difficult_words"     : textstat.difficult_words(text),
        "avg_words_per_sent"  : round(textstat.avg_sentence_length(text), 2),
        "flesch_reading_ease" : round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "gunning_fog"         : round(textstat.gunning_fog(text), 2),
        "smog_index"          : round(textstat.smog_index(text), 2),
        "ari_score"           : round(textstat.automated_readability_index(text), 2)
    }