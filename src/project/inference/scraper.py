from newspaper import Article, Config

# Set up a custom user agent to avoid blocking
config = Config()
config.browser_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

def scrape_article(url: str) -> str:
    article = Article(url, config=config)
    article.download()
    article.parse()
    return article.text
if __name__ == "__main__":
    test_url = "https://finance.yahoo.com/news/5-things-know-stock-market-131502868.html"
    content = scrape_article(test_url)
    print(content[:500])  # Print the first 500 characters of the article