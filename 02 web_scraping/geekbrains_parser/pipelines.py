from itemadapter import ItemAdapter
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from geekbrains_parser.spiders.gb_ru import GbRuSpider
import csv
import os


class GeekbrainsParserPipeline:
    def __init__(self):
        client = MongoClient('127.0.0.1', 27017)
        self.mongo_base = client.GeekBrains

    def process_item(self, item, spider):
        collection = self.mongo_base[spider.name]
        try:
            collection.insert_one(item)
            print(f"Запись о специальности {item['name']} добавлена")
        except DuplicateKeyError:
            collection.replace_one({'_id': item['_id']}, item)
            print(f"Дубль! Запись о специальности {item['name']} обновлена")
        return item


class CSVPipeline(object):
    def __init__(self):
        self.file = f'{GbRuSpider.name}.csv'
        if self.file not in os.getcwd():
            open(self.file, "w").close()
        with open(self.file, 'r', newline='') as csv_file:
            self.tmp_data = csv.DictReader(csv_file).fieldnames
        self.csv_file = open(self.file, 'a', newline='', encoding='UTF-8')

    def process_item(self, item, spider):
        columns = item.fields.keys()
        data = csv.DictWriter(self.csv_file, columns)
        if not self.tmp_data:
            data.writeheader()
            self.tmp_data = True
        data.writerow(item)
        return item

    def __del__(self):
        self.csv_file.close()
