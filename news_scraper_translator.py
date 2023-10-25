import requests
from bs4 import BeautifulSoup
from googletrans import Translator
from typing import Optional

class NewsScraper:
    def __init__(self, url: str) -> None:
        """
        Initialise the NewsScraper class.

        Args:
            url (str): URL of the news article to be scraped.

        """
        self.url = url
        self.title: Optional[str] = None
        self.content: Optional[str] = None
        self.soup: Optional[BeautifulSoup] = None

    def fetch_data(self) -> None:
        """
        Fetch the web page content and initialise the BeautifulSoup object.

        """
        # Fetch the web page using requests
        response = requests.get(self.url)
        # Parse the page content with BeautifulSoup
        self.soup = BeautifulSoup(response.content, "html.parser")

    def extract_title(self) -> Optional[str]:
        """
        Extract the title of the news article.

        Returns:
            Optional[str]: The extracted title if found, otherwise None.

        """
        if self.soup:
            self.title = self.soup.find("title").text.strip()
        return self.title

    def extract_content(self) -> Optional[str]:
        """
        Extract the main content of the news article.

        Returns:
            Optional[str]: The extracted content if found, otherwise None.

        """
        if self.soup:
            self.content = self.soup.find("div", {"class": "caas-body"}).text.strip()
        return self.content

class TextTranslator:
    def __init__(self) -> None:
        """
        Initialise the TextTranslator class.

        """
        self.translator = Translator()

    def translate(self, text: str) -> str:
        """
        Translate the provided text into English.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text.

        """
        return self.translator.translate(text).text

if __name__ == "__main__":
    # Define the URL for scraping
    url = "https://tw.news.yahoo.com/%E5%A4%A9%E6%B0%A3-%E6%B8%85%E6%98%8E-%E4%B8%8B%E9%9B%A8-233008705.html"
    # Initialise and fetch news details using NewsScraper
    news_scraper = NewsScraper(url)
    news_scraper.fetch_data()

    title = news_scraper.extract_title()
    content = news_scraper.extract_content()
    print("Title:", title)
    print("Content:", content)

    # Translate the extracted title into English
    translator = TextTranslator()
    translated_title = translator.translate(title)
    print("Translated title:", translated_title)
