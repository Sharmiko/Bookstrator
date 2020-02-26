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

    
    def __auto_detect_font(self, blocks: List) -> Dict:
        """Functiom that autodetects font-sizes on the given Page

        Parameters:
            blocks (List):

        Returns:
            Dict: dictionary of font-sizes and their corresponding
                  text frequency
        """
        font_dict = dict()

        for block in blocks:
            lines: Dict = block.get("lines")

            for line in lines:
                spans: Dict = line.get("spans")[0]

                font_count: int = font_dict.get(spans.get("size"), 0)
                current_count: int = len(spans.get("text"))

                font_dict[spans.get("size")] = font_count + current_count

        return font_dict


    def __clean_content(self, blocks: List) -> Dict:
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

        body = ""
        title = ""

        font_dict: Dict = self.__auto_detect_font(blocks)
        body_font: int = max(font_dict, key=font_dict.get)
        font_dict[body_font] = -1
        title_font: int = max(font_dict, key=font_dict.get)


        for block in blocks:
            lines: Dict = block.get("lines")

            for line in lines:
                spans = line.get("spans")[0]

                # extract content of body
                if int(spans.get("size")) == body_font:
                    body += spans.get("text")
                elif int(spans.get("size")) == title_font:
                    title += spans.get("text")

        
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

        return self.__clean_content(blocks)


    def to_csv(self, file_name: str):
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
        df.to_csv(file_name, index=False)