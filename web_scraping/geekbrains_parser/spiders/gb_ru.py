import scrapy
from scrapy.http import HtmlResponse
import requests
from geekbrains_parser.items import GeekbrainsParserItem
from dotenv import load_dotenv, find_dotenv
import os
from copy import deepcopy


class GbRuSpider(scrapy.Spider):
    load_dotenv(find_dotenv())
    LOGIN = os.environ.get('login')
    PASSWORD = os.environ.get('password')
    name = 'gb_ru'
    allowed_domains = ['gb.ru']
    start_urls = ['https://gb.ru/login/']
    login_link = 'https://gb.ru/login'
    client = requests.session()
    login_m = LOGIN
    password = PASSWORD

    def parse(self, response: HtmlResponse):
        post_hashe = response.text[response.text.find('csrf-token')+21:response.text.find('csrf-token')+109]
        yield scrapy.FormRequest(
            'https://gb.ru/login/',
            method='POST',
            callback=self.login,
            formdata={
                'utf8': '✓',
                'authenticity_token': post_hashe,
                'user[email]': self.login_m,
                'user[password]': self.password,
                'user[remember_me]': '0'})

    def login(self, response: HtmlResponse):
        if response.text.find('Моё обучение'):
            big_block = response.xpath("//div[contains(@class, 'columns_xxl_4')]//a/@href").getall()
            for block in big_block:
                yield response.follow(
                    block,
                    callback=self.studying_programs)

    def studying_programs(self, response: HtmlResponse):
        xpath_name = "//div[@class='paragraph new-d w-richtext']/p/text()"
        xpath_description = "//div[@class='collection-list new-d w-dyn-items']//a/@href"
        xpath_link = "//div[@class='collection-list new-d w-dyn-items']//div[@class='product_title new-d']/text()"
        all_descriptions = [i for i in response.xpath(
            f'{xpath_name}|{xpath_description}|{xpath_link}'
        ).getall() if i != '\u200d']
        for num, card in enumerate(all_descriptions):
            if num % 3 == 0:
                description = {'name': all_descriptions[num], 'descr': all_descriptions[num+1]}
                link = all_descriptions[num+2]
                yield response.follow(
                    link,
                    callback=self.description_of_the_training_program,
                    cb_kwargs={'description': deepcopy(description)})

    def description_of_the_training_program(self, response: HtmlResponse, description):
        name = description['name']
        text = description['descr']
        link = response.url
        _id = response.url
        # price = ...
        yield GeekbrainsParserItem(name=name, text=text, link=link, _id=_id)
