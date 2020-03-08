import spacy 
import pandas as pd 

nlp = spacy.load("en_core_web_sm")
to_keep = ["NOUN", "ADV", "VERB", "PROPN", "NUM"]

def clean_text(text):
    doc = nlp(text)
    new_text = []
    for token in doc:    
        if token.pos_ in to_keep:
            new_text.append(token.lemma_)
    
    return " ".join(new_text)


df = pd.read_csv("out_sum.csv")
sample = df["body"][0]
print("ORIGINAL: \n", sample)
print("")
print("REDUCES: \n", clean_text(sample))