# modules/preprocessing.py
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocessing_row(text):
    custom_stopwords = [
        "user",
        "recommends",
        "suggests",
        "feature",
        "request",
        "issue",
        "improvement",
        "problem",
        "question",
        "suggestion",
        "feedback",
        "integrate",
        "wants",
        "needs",
        "difficulty",
        "error",
        "bug",
        "option",
    ]

    doc = nlp(text)
    cleaned_words = []
    
    for token in doc:
        # Remove numbers and verbs
        if not (token.is_stop or token.is_punct or token.like_num or token.pos_ == "VERB") and token.lemma_.lower() not in custom_stopwords:
            cleaned_words.append(token.lemma_)
    
    cleaned_text = " ".join(cleaned_words)
    return cleaned_text

def preprocess_text(input_file):

    # Read the input CSV
    data = pd.read_csv(input_file, delimiter="|")

    # Combine "Name" and "Reason" columns for text analysis
    data["text"] = data["name"] + " " + data["reasson"]

    # Apply custom preprocessing to the text data
    data["text"] = data["text"].apply(preprocessing_row)
    
    return data
