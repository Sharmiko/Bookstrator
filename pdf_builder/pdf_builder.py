import argparse
import pdfkit
import pandas as pd
from markdown import markdown


def generate_html(html_path: str, title: str, image_path: str, 
    body :str) -> str:
    # open htnl tempalte
    with open(html_path, 'r') as f:
        html_text = f.read()
    
    # replace placeholder with corresponding information
    html_text = html_text.replace("[title]", title)
    html_text = html_text.replace("[img]", image_path)
    html_text = html_text.replace("[body]", body)

    # convert html text into markdown
    html_text = markdown(html_text, output_format='html4')

    return html_text


def html_to_pdf(html_text: str, pdf_name: str) -> None:
    # function that converts html text into PDF file
    pdfkit.from_string(html_text, pdf_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, help="Location of input csv file")
    parser.add_argument("--html_template", type=str, help="Location of input html file")
    parser.add_argument("--output_name", type=str, help="starting page of the range")

    opt = parser.parse_args()

    df = pd.read_csv(opt.csv_file)

    html_page = ""
    for _, fable in df.iterrows():
        html_page += generate_html(opt.html_template, fable["title"],
            fable["img_path"], "\t" + fable["body"])
        html_page += "</br>"*12

    html_to_pdf(html_page, opt.output_name)