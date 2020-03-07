import argparse

import pandas as  pd 
from tqdm import tqdm 

from Summarizers import Summarizers

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Location of input file")
    parser.add_argument("--output_file", type=str, help="Location of output file")

    opt = parser.parse_args()

    df = pd.read_csv(opt.input_file)
    fables = df["body"]

    summarizer = Summarizers()

    summarized_bert = []
    total = len(fables)
    i = 1 
    print("Summarizing...")
    for fable in tqdm(fables):
        print(f"Fable {i}/{total}")
        i += 1
        summarized_bert.append(summarizer.summarize(fable, summarizer="bert"))
    df["bert"] = summarized_bert
    print("Done!")
    
    df.to_csv(opt.output_file, index=False)


