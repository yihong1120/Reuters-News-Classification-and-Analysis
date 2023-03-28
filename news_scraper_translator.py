import requests
from bs4 import BeautifulSoup
from googletrans import Translator

class NewsScraper:
    def __init__(self, url):
        self.url = url
        self.title = None
        self.content = None
        self.soup = None

    def fetch_data(self):
        response = requests.get(self.url)
        self.soup = BeautifulSoup(response.content, "html.parser")

    def extract_title(self):
        if self.soup:
            self.title = self.soup.find("title").text.strip()
        return self.title

    def extract_content(self):
        if self.soup:
            self.content = self.soup.find("div", {"class": "caas-body"}).text.strip()
        return self.content

class TextTranslator:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text):
        return self.translator.translate(text).text

# Usage
if __name__ == "__main__"
    url = "https://tw.news.yahoo.com/%E5%A4%A9%E6%B0%A3-%E6%B8%85%E6%98%8E-%E4%B8%8B%E9%9B%A8-233008705.html"
    news_scraper = NewsScraper(url)
    news_scraper.fetch_data()

    title = news_scraper.extract_title()
    content = news_scraper.extract_content()
    print("標題：", title)
    print("內文：", content)

    translator = TextTranslator()
    translated_title = translator.translate(title)
    print("Translated title:", translated_title)
