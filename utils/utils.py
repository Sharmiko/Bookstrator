import spacy 
import pandas as pd 

nlp = spacy.load("en_core_web_lg")
#to_keep = ["NOUN", "ADV", "VERB", "PROPN", "NUM"]
to_keep = ["NOUN"]
def clean_text(text):
    doc = nlp(text)
    new_text = []
    # for token in doc:    
    #     if token.pos_ in to_keep:
    #         print(token, token.pos_, token.lemma_, token.label_)
    #         new_text.append(token.lemma_)
    
    to_discrad = ["DATE"]
    for token in doc.ents:
        print(token, token.label_)
        if token.label_ not in to_discrad:
            new_text.append(token)
    
    return new_text


df = pd.read_csv("out_sum.csv")
sample = df["body"][2].lower()
print("ORIGINAL: \n", sample)
print("")
print("REDUCED: \n", set(clean_text(sample)))