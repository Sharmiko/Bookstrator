import argparse

import spacy 
import pandas as pd
from tqdm import tqdm
from azure.ai.textanalytics import TextAnalyticsClient, TextAnalyticsApiKeyCredential

key = "2df9c005a07d4960a77e2e5910da78d7"
endpoint = "https://bookstraotrtext.cognitiveservices.azure.com/"


def authenticate_client():
    ta_credential = TextAnalyticsApiKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

client = authenticate_client()

def sentiment_analysis_example(client, document):

    response = client.analyze_sentiment(inputs=document)[0]
    print("Document Sentiment: {}".format(response.sentiment))
    print("Overall scores: positive={0:.2f}; neutral={1:.2f}; negative={2:.2f} \n".format(
        response.confidence_scores.positive,
        response.confidence_scores.neutral,
        response.confidence_scores.negative,
    ))
    for idx, sentence in enumerate(response.sentences):
        print("[Length: {}]".format(sentence.grapheme_length))
        print("Sentence {} sentiment: {}".format(idx+1, sentence.sentiment))
        print("Sentence score:\nPositive={0:.2f}\nNeutral={1:.2f}\nNegative={2:.2f}\n".format(
            sentence.confidence_scores.positive,
            sentence.confidence_scores.neutral,
            sentence.confidence_scores.negative,
        ))

            

def entity_recognition_example(client, document):
    try:
        result = client.recognize_entities(inputs= document)[0]

        print("Named Entities:\n")
        for entity in result.entities:
            print("\tText: \t", entity.text, "\tConfidence Score: \t", round(entity.score, 2), "\n")

    except Exception as err:
        print("Encountered exception. {}".format(err))


def key_phrase_extraction(client, document):
    try:
        response = client.extract_key_phrases(inputs= document)[0]

        if not response.is_error:
            return [phrase for phrase in response.key_phrases]
        else:
            print(response.id, response.error)

    except Exception as err:
        print("Encountered exception. {}".format(err))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Location of input file")
    parser.add_argument("--output_file", type=str, help="Location of output file")

    opt = parser.parse_args()

    df = pd.read_csv(opt.input_file)
    fables = df["body"]

    key_phrases = list()
    print("Extracting key phrases...")
    for fable in tqdm(fables):
        phrase = key_phrase_extraction(client, [fable])
        phrase = " ".join(phrase)
        key_phrases.append(phrase)
    print("Done!")
    
    print()
    print("Sample Output:")
    print(key_phrases[0] + "...")

    out = pd.DataFrame({"fables": fables, "key_phrases": key_phrases})
    out.to_csv(opt.output_file, index=False)