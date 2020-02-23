from PDFReader import PDFReader

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Location of input pdf file")
    parser.add_argument("--output_file", type=str, help="Location of output file to write into")
    parser.add_argument("--title_size", type=int, default=20, help="Font size of the title on the content page")
    parser.add_argument("--body_size", type=int, default=10, help="Font size of the body on the content page")
    parser.add_argument("--start_page", type=int, help="starting page of the range")
    parser.add_argument("--end_page", type=int, help="ending page of the range")

    opt = parser.parse_args()

    pdf_reader = PDFReader(opt.input_file)

    if opt.title_size != 20:
        pdf_reader.title_size = opt.title_size

    if opt.body_size != 10:
        pdf_reader.body_size = opt.body_size

    pdf_reader.to_csv(opt.output_file, opt.start_page, opt.end_page)
    

