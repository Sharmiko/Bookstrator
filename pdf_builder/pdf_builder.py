import pandas as pd
from markdown import markdown
import pdfkit


def generate_html(html_path, title, image_path, body):
    with open(html_path, 'r') as f:
        html_text = f.read()
    
    html_text = html_text.replace("[title]", title)
    html_text = html_text.replace("[img]", image_path)
    html_text = html_text.replace("[body]", body)

    html_text = markdown(html_text, output_format='html4')

    return html_text


def html_to_pdf(html_text, pdf_name):
    pdfkit.from_string(html_text, pdf_name)

if __name__ == "__main__":

    df = pd.read_csv("fables.csv")
    html_page = ""
    for _, fable in df.iterrows():
        html_page += generate_html("pdf_template.html", fable["title"],
            fable["img_path"], "\t" + fable["body"])
        html_page += "</br>"*12

    html_to_pdf(html_page, "illustrated_fables.pdf")