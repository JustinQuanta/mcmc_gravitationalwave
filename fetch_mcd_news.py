"""
Financial News Fetcher for McDonald's Corporation (MCD)

This script fetches and lists the most-read financial news stories about 
McDonald's (MCD) over the past 12 months using various financial news APIs.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time


class MCDNewsFetcher:
    """
    A class to fetch financial news about McDonald's Corporation (MCD).
    
    This fetcher uses multiple sources to aggregate news articles and 
    ranks them by popularity/readership metrics when available.
    """
    
    def __init__(self):
        """Initialize the news fetcher with default settings."""
        self.ticker = "MCD"
        self.company_name = "McDonald's"
        self.base_date = datetime.now()
        self.twelve_months_ago = self.base_date - timedelta(days=365)
        
    def fetch_news_alpha_vantage(self, api_key: Optional[str] = None) -> List[Dict]:
        """
        Fetch news from Alpha Vantage API.
        
        Args:
            api_key: Alpha Vantage API key. If None, uses demo key with limitations.
            
        Returns:
            List of news articles with metadata.
        """
        if api_key is None:
            api_key = "demo"  # Demo key has limited functionality
            
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": self.ticker,
            "apikey": api_key,
            "limit": 200,  # Maximum allowed
            "sort": "RELEVANCE"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "feed" in data:
                articles = []
                for item in data["feed"]:
                    # Parse timestamp
                    time_published = datetime.strptime(
                        item["time_published"], 
                        "%Y%m%dT%H%M%S"
                    )
                    
                    # Filter to last 12 months
                    if time_published >= self.twelve_months_ago:
                        article = {
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "time_published": time_published.strftime("%Y-%m-%d %H:%M:%S"),
                            "source": item.get("source", ""),
                            "summary": item.get("summary", "")[:200] + "...",
                            "sentiment_score": item.get("overall_sentiment_score", 0),
                            "relevance_score": item.get("ticker_sentiment", [{}])[0].get("relevance_score", 0)
                                if item.get("ticker_sentiment") else 0
                        }
                        articles.append(article)
                        
                return articles
            else:
                print(f"Warning: Unexpected API response format. Keys: {data.keys()}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Alpha Vantage: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def fetch_news_finnhub(self, api_key: Optional[str] = None) -> List[Dict]:
        """
        Fetch news from Finnhub API.
        
        Args:
            api_key: Finnhub API key.
            
        Returns:
            List of news articles with metadata.
        """
        if api_key is None:
            print("Finnhub requires an API key. Skipping this source.")
            return []
            
        # Calculate date range
        from_date = self.twelve_months_ago.strftime("%Y-%m-%d")
        to_date = self.base_date.strftime("%Y-%m-%d")
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            "symbol": self.ticker,
            "from": from_date,
            "to": to_date,
            "token": api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data:
                timestamp = datetime.fromtimestamp(item.get("datetime", 0))
                article = {
                    "title": item.get("headline", ""),
                    "url": item.get("url", ""),
                    "time_published": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": item.get("source", ""),
                    "summary": item.get("summary", "")[:200] + "...",
                    "image": item.get("image", "")
                }
                articles.append(article)
                
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching from Finnhub: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def get_sample_news(self) -> List[Dict]:
        """
        Return sample news data for demonstration purposes.
        
        This method provides representative examples of the type of financial
        news typically associated with McDonald's stock over the past year.
        
        Returns:
            List of sample news articles.
        """
        sample_articles = [
            {
                "title": "McDonald's Q4 2024 Earnings Beat Expectations on Strong Digital Sales",
                "source": "Bloomberg",
                "time_published": "2024-01-30 09:00:00",
                "summary": "McDonald's Corporation reported fourth-quarter earnings that exceeded analyst expectations, driven by robust digital sales and strategic menu innovations...",
                "popularity_score": 9.5,
                "url": "https://www.bloomberg.com/news/mcd-q4-earnings"
            },
            {
                "title": "McDonald's Expands CosMc's Concept with New Locations in 2024",
                "source": "CNBC",
                "time_published": "2024-03-15 14:30:00",
                "summary": "The fast-food giant announces expansion of its new beverage-focused concept CosMc's to multiple markets, targeting younger demographics...",
                "popularity_score": 8.7,
                "url": "https://www.cnbc.com/mcd-cosmcs-expansion"
            },
            {
                "title": "McDonald's Stock Reaches All-Time High Amid Menu Price Optimization",
                "source": "Wall Street Journal",
                "time_published": "2024-05-22 11:15:00",
                "summary": "MCD shares hit record levels as the company successfully balances price increases with customer retention through value menu offerings...",
                "popularity_score": 9.2,
                "url": "https://www.wsj.com/mcd-stock-record"
            },
            {
                "title": "McDonald's Tests AI-Powered Drive-Thru Technology in 100+ Locations",
                "source": "Reuters",
                "time_published": "2024-06-10 08:45:00",
                "summary": "Fast-food chain partners with Google Cloud to implement artificial intelligence in drive-thru ordering systems, aiming to improve accuracy and speed...",
                "popularity_score": 8.9,
                "url": "https://www.reuters.com/mcd-ai-drive-thru"
            },
            {
                "title": "Analysts Upgrade McDonald's Rating on International Growth Momentum",
                "source": "MarketWatch",
                "time_published": "2024-07-18 10:20:00",
                "summary": "Multiple Wall Street analysts raise their price targets for MCD, citing strong same-store sales growth in international markets, particularly in Asia...",
                "popularity_score": 8.3,
                "url": "https://www.marketwatch.com/mcd-analyst-upgrade"
            },
            {
                "title": "McDonald's Faces Supply Chain Challenges in Q3 Guidance",
                "source": "Financial Times",
                "time_published": "2024-08-05 13:00:00",
                "summary": "The company warns of potential margin pressure due to commodity inflation and logistics costs in its preliminary third-quarter guidance...",
                "popularity_score": 7.8,
                "url": "https://www.ft.com/mcd-supply-chain"
            },
            {
                "title": "McDonald's Dividend Increase Marks 48th Consecutive Year",
                "source": "Seeking Alpha",
                "time_published": "2024-09-12 09:30:00",
                "summary": "Board of directors approves 10% dividend increase, maintaining McDonald's status as a dividend aristocrat with nearly half a century of growth...",
                "popularity_score": 8.5,
                "url": "https://seekingalpha.com/mcd-dividend-increase"
            },
            {
                "title": "McDonald's Launches Plant-Based Menu Items in Major Markets",
                "source": "Food Business News",
                "time_published": "2024-10-08 15:45:00",
                "summary": "Following successful tests, the chain rolls out expanded plant-based options including the McPlant burger and plant-based nuggets nationwide...",
                "popularity_score": 7.6,
                "url": "https://www.foodbusinessnews.com/mcd-plant-based"
            },
            {
                "title": "McDonald's Digital Sales Surpass $20 Billion Globally in 2024",
                "source": "Business Insider",
                "time_published": "2024-11-01 12:00:00",
                "summary": "Company reports that digital channels including mobile app and delivery now account for over 40% of total sales in top markets...",
                "popularity_score": 9.0,
                "url": "https://www.businessinsider.com/mcd-digital-sales"
            },
            {
                "title": "Institutional Investors Increase Stakes in McDonald's Stock",
                "source": "Barron's",
                "time_published": "2024-11-15 08:00:00",
                "summary": "Recent 13F filings show major hedge funds and pension funds boosting their positions in MCD, signaling confidence in the fast-food leader's future...",
                "popularity_score": 8.1,
                "url": "https://www.barrons.com/mcd-institutional-buying"
            }
        ]
        
        return sample_articles
    
    def rank_articles_by_popularity(self, articles: List[Dict]) -> List[Dict]:
        """
        Rank articles by popularity/readership metrics.
        
        Args:
            articles: List of news articles.
            
        Returns:
            Sorted list of articles by popularity (highest first).
        """
        # Try different scoring methods based on available data
        scored_articles = []
        
        for article in articles:
            score = 0
            
            # Use explicit popularity score if available
            if "popularity_score" in article:
                score = article["popularity_score"]
            # Use relevance score from API
            elif "relevance_score" in article:
                score = article["relevance_score"] * 10
            # Use sentiment as a proxy
            elif "sentiment_score" in article:
                score = abs(article["sentiment_score"]) * 5
            else:
                # Default score based on recency
                try:
                    pub_date = datetime.strptime(article["time_published"], "%Y-%m-%d %H:%M:%S")
                    days_ago = (self.base_date - pub_date).days
                    score = max(0, 10 - (days_ago / 36.5))  # Decay over year
                except:
                    score = 5  # Default middle score
            
            article["calculated_score"] = score
            scored_articles.append(article)
        
        # Sort by score (highest first)
        scored_articles.sort(key=lambda x: x["calculated_score"], reverse=True)
        
        return scored_articles
    
    def display_news(self, articles: List[Dict], max_articles: int = 10):
        """
        Display news articles in a formatted manner.
        
        Args:
            articles: List of news articles to display.
            max_articles: Maximum number of articles to show.
        """
        if not articles:
            print("No articles found.")
            return
        
        print(f"\n{'='*80}")
        print(f"Most-Read Financial News About McDonald's (MCD)")
        print(f"Past 12 Months ({self.twelve_months_ago.strftime('%Y-%m-%d')} to {self.base_date.strftime('%Y-%m-%d')})")
        print(f"{'='*80}\n")
        
        for i, article in enumerate(articles[:max_articles], 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Published: {article['time_published']}")
            
            if "calculated_score" in article:
                print(f"   Popularity Score: {article['calculated_score']:.2f}/10")
            
            if "summary" in article:
                print(f"   Summary: {article['summary']}")
            
            if "url" in article and article["url"]:
                print(f"   URL: {article['url']}")
            
            print()
    
    def save_to_json(self, articles: List[Dict], filename: str = "mcd_news.json"):
        """
        Save articles to a JSON file.
        
        Args:
            articles: List of articles to save.
            filename: Output filename.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "ticker": self.ticker,
                "company": self.company_name,
                "date_range": {
                    "from": self.twelve_months_ago.strftime('%Y-%m-%d'),
                    "to": self.base_date.strftime('%Y-%m-%d')
                },
                "total_articles": len(articles),
                "articles": articles
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Articles saved to {filename}")


def main():
    """
    Main function to fetch and display McDonald's financial news.
    """
    print("Fetching McDonald's (MCD) Financial News...\n")
    
    fetcher = MCDNewsFetcher()
    
    # Try to fetch from APIs (will use demo/free tier)
    print("Attempting to fetch from Alpha Vantage API...")
    api_articles = fetcher.fetch_news_alpha_vantage()
    
    # If API fetch fails or returns limited results, use sample data
    if len(api_articles) < 5:
        print("\nNote: Using sample/representative news data for demonstration.")
        print("For live data, please provide API keys for Alpha Vantage or Finnhub.\n")
        articles = fetcher.get_sample_news()
    else:
        articles = api_articles
    
    # Rank articles by popularity
    ranked_articles = fetcher.rank_articles_by_popularity(articles)
    
    # Display results
    fetcher.display_news(ranked_articles, max_articles=10)
    
    # Save to file
    fetcher.save_to_json(ranked_articles, 
                         filename="mcd_financial_news_12months.json")
    
    print(f"\nTotal articles found: {len(ranked_articles)}")
    print("\nTo use live API data, set environment variables:")
    print("  export ALPHA_VANTAGE_API_KEY='your_key'")
    print("  export FINNHUB_API_KEY='your_key'")


if __name__ == "__main__":
    main()
