import fitz 
import pandas as pd 

from typing import List, Dict

class PDFReader(object):
    
    def __init__(self, pdf_file: str):
        """ Open given pdf file and read content from it
        
        Parameters:
            pdf_file (str): Location of the pdf file
        """
        self.pdf = fitz.open(pdf_file)


    @property
    def pages(self) -> int:
        """ Return number of pages that the document contains

        Returns:
            int: number of pages
        """
        return self.pdf.pageCount

    def clean_content(self, blocks: List, title_size:int=20, 
                      body_size:int=10) -> Dict:
        """ Function that extracts textual information based on 
            font meta-data.

            For example:
                If font-size is 20px, then text
                is considered to be a title, if it's 10px
                it is considered to a body content and if 
                font-size is 7px it is considered to be a 
                footer and it's ignored.

        Parameters:
            blocks (list): 

            title_size (int): font-size of the title on the content page

            body_size (int): font-size of the body on the content page

        Returns:
            Dict: dictionary of the title and the body
        """

        title = ""
        body = ""

        for block in blocks:
            lines = block.get("lines")

            for line in lines:
                spans = line.get("spans")[0]

                # extract title
                if int(spans.get("size")) == title_size:
                    title += spans.get("text")
                # extract content of body
                elif int(spans.get("size")) == body_size:
                    body += spans.get("text")

        return {"title": title, "body": body}

    def getPageContent(self, page: int) -> Dict:
        """ Return extracted text from the requested page
        
        Parameters:
            page (int): number of page to extract content from

        Returns:
            Dict: returns dictionary from the given page, containing
                  the title and the body 
        """

        load_page = self.pdf.loadPage(page)
        blocks = load_page.getDisplayList().getTextPage().extractDICT().get("blocks")
        return self.clean_content(blocks)

    def to_csv(self, file_name: str, start_page: int, end_page: int):
        """ Function that converts contents of the given range of pages
            into pandas DataFrame csv file 
            User must provide starting and ending page of the pdf file,
            because range may vary from one file to another

        Parameters:
            file_name (str): path to file to save content into

            start_page (int): starting page of the range

            end_page (int): ending page of the range 
        """
        content_dict: List[Dict] = list()
        for page in range(start_page, end_page):
            temp = {}
            content = self.getPageContent(page)
            title, body = content.get("title"), content.get("body")

            # case when text is continued on other page
            # in this case append current body to previous body
            # to get full text
            if title == '':
                content_dict[len(content_dict) - 1]["body"] += body
            else:
                temp["title"] = title 
                temp["body"] = body
                content_dict.append(temp)
        
        df = pd.DataFrame(content_dict)
        df.to_csv("out.csv", index=False)