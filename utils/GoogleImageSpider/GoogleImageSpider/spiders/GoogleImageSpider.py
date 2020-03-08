import scrapy 
import urlparse 

class GoogleImageSpider(scrapy.Spider):
    name = 'google-spider'
    collection_name = 'google'

    search_tempalte = 'https://google.com/search?q={}&tbm=isch'

    def parse(self, response):
        pass