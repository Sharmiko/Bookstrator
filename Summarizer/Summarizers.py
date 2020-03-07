from summarizer import Summarizer

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.web_scraping import WebScraping
from pysummarization.abstractabledoc.std_abstractor import StdAbstractor
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

class Summarizers(object):
    """Summarizers class that supports multiple models
    for text summarization.

    Supported models:

        Bert-extractive-summarizer

        Pysummarization

        Pysummarization-skip-gram

    """

    def __init__(self):
        self.bert_summarizer = Summarizer()

    def summarize(self, text: str, summarizer: str="bert") -> str:
        """Function that summarizes text based on chosen model

        Parameters:
            text (str): input string to summarize
            summarizer (str): summarizer to use
                supported summarizers: "bert" -> Bert-extractive-summarizer
                                       "pysum" -> Pysummarization
                                       "pysum-skip-gram" -> Pysummarization-skip-gram
        
        Returns:
            str: summarized text
        """
        summarizers = {
            "bert": self.bert_summarizer,
            "pysum": self.pysummarization,
            "pysum-skip-gram": self.pysummarization_skip_gram
        }

        summarizer = summarizers.get(summarizer, "N/A")
        
        # If invalid summarizer is supplied raise ValueError
        if summarizer == "N/A":
            return self.summarizer_error

        return summarizer(text)
        

    def bert_summarizer(self, text: str, min_length: int=60, max_length: int=600) -> str:
        """Summarizer Based on Bert-extractive summarizer

        Parameters:
            text (str): text to summarize
            min_length (int): minimum length of summarized text
            max_length (int): maximum lenght of summarized text

        Returns:
            str: summarized text
        """
        text = ''.join(self.bert_summarizer(text, max_length=max_length))

        return text 


    def pysummarization(self, text: str, max_sentences: int=5) -> str:
        """Summarir based on pysummerization

        Parameters:
            text (str): text to summarize
            max_sentences (int): maximum number of sentences

        Returns:
            str: summarized text
        """

        auto_abstractor = AutoAbstractor()
        auto_abstractor.tokenizable_doc = SimpleTokenizer()
        auto_abstractor.delimiter_list = [".", "\n"]
        abstractable_doc = TopNRankAbstractor()
        result_dict = auto_abstractor.summarize(text, abstractable_doc)

        sentences = result_dict["summarize_result"]
        indices = {}
        for i, sentence in enumerate(sentences):
            indices[sentence] = i

        def sort_key(sentence):
            index = indices[sentence]
            score = result_dict['scoring_data'][index]
            return score[1]

        sorted_sentences = sorted(sentences, key=sort_key)

        return ' '.join(sorted_sentences)


    def pysummarization_skip_gram(self):
        pass 

    @property
    def summarizer_error(self):
        """ Raise ValueError if invalid summarizer model name is specified
        """
        raise ValueError("Invalid input. Model not found!")
    
