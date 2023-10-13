import scrapy


class GeekbrainsParserItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    text = scrapy.Field()
    link = scrapy.Field()
    _id = scrapy.Field()
