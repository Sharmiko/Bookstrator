from PDFReader import PDFReader

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Location of input pdf file")
    parser.add_argument("--output_file", type=str, help="Location of output file to write into")
    parser.add_argument("--start_page", type=int, help="starting page of the range")
    parser.add_argument("--end_page", type=int, help="ending page of the range")

    opt = parser.parse_args()

    pdf_reader = PDFReader(opt.input_file)

    pdf_reader.to_csv(opt.output_file, opt.start_page, opt.end_page)