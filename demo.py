from news import NewsScraper, TextTranslator
from reuters_model import ReutersModel, ReutersTrainer, ReutersPredictor

class ModelTrainer:
    def __init__(self, reuters_model):
        self.reuters_model = reuters_model

    def train_model(self):
        trainer = ReutersTrainer(self.reuters_model)
        history, results = trainer.train_and_evaluate()
        return history, results

class NewsAnalyzer:
    def __init__(self, url, reuters_model):
        self.url = url
        self.reuters_model = reuters_model
        self.news_scraper = NewsScraper(self.url)
        self.translator = TextTranslator()
        self.predictor = ReutersPredictor(self.reuters_model)

    def analyze_news(self):
        self.news_scraper.fetch_data()
        title = self.news_scraper.extract_title()
        content = self.news_scraper.extract_content()
        translated_text = self.translator.translate(content)
        predicted_label = self.predictor.predict(translated_text)
        return predicted_label

# Usage
if __name__ == "__main__"
    reuters_model = ReutersModel()
    model_trainer = ModelTrainer(reuters_model)
    history, results = model_trainer.train_model()

    news_analyzer = NewsAnalyzer("https://tw.news.yahoo.com/%E5%A4%A9%E6%B0%A3-%E6%B8%85%E6%98%8E-%E4%B8%8B%E9%9B%A8-233008705.html", reuters_model)
    predicted_label = news_analyzer.analyze_news()
    print("Predicted label:", predicted_label)
