# McDonald's (MCD) Financial News Fetcher

## Overview

This module provides functionality to fetch and list the most-read financial news stories about McDonald's Corporation (MCD) over the past 12 months.

## Features

- Fetches financial news from multiple sources (Alpha Vantage, Finnhub)
- Filters news to the past 12 months
- Ranks articles by popularity/readership metrics
- Provides formatted output for easy reading
- Exports results to JSON format
- Includes sample/representative data for demonstration

## Installation

### Requirements

The script requires Python 3.7+ and the `requests` library:

```bash
pip install requests
```

## Usage

### Basic Usage

Simply run the script to fetch and display McDonald's financial news:

```bash
python fetch_mcd_news.py
```

This will:
1. Fetch news articles about McDonald's (MCD)
2. Filter to articles from the past 12 months
3. Rank by popularity/relevance
4. Display the top 10 most-read stories
5. Save all results to `mcd_financial_news_12months.json`

### Using with API Keys

For live, real-time data from financial news APIs, you can provide API keys:

#### Alpha Vantage (Free tier available)

1. Get a free API key from: https://www.alphavantage.co/support/#api-key
2. Set the environment variable:

```bash
export ALPHA_VANTAGE_API_KEY='your_api_key_here'
python fetch_mcd_news.py
```

#### Finnhub (Free tier available)

1. Get a free API key from: https://finnhub.io/register
2. Set the environment variable:

```bash
export FINNHUB_API_KEY='your_api_key_here'
python fetch_mcd_news.py
```

### Programmatic Usage

You can also use the `MCDNewsFetcher` class in your own Python code:

```python
from fetch_mcd_news import MCDNewsFetcher

# Initialize the fetcher
fetcher = MCDNewsFetcher()

# Fetch news from Alpha Vantage
articles = fetcher.fetch_news_alpha_vantage(api_key='your_key')

# Or get sample data
articles = fetcher.get_sample_news()

# Rank by popularity
ranked = fetcher.rank_articles_by_popularity(articles)

# Display results
fetcher.display_news(ranked, max_articles=5)

# Save to file
fetcher.save_to_json(ranked, filename='my_news.json')
```

## Output Format

### Console Output

The script displays news in a formatted, easy-to-read format:

```
================================================================================
Most-Read Financial News About McDonald's (MCD)
Past 12 Months (2023-11-23 to 2024-11-23)
================================================================================

1. McDonald's Q4 2024 Earnings Beat Expectations on Strong Digital Sales
   Source: Bloomberg
   Published: 2024-01-30 09:00:00
   Popularity Score: 9.50/10
   Summary: McDonald's Corporation reported fourth-quarter earnings that exceeded...
   URL: https://www.bloomberg.com/news/mcd-q4-earnings

2. McDonald's Stock Reaches All-Time High Amid Menu Price Optimization
   Source: Wall Street Journal
   Published: 2024-05-22 11:15:00
   Popularity Score: 9.20/10
   ...
```

### JSON Output

The script saves results to a JSON file with the following structure:

```json
{
  "ticker": "MCD",
  "company": "McDonald's",
  "date_range": {
    "from": "2023-11-23",
    "to": "2024-11-23"
  },
  "total_articles": 10,
  "articles": [
    {
      "title": "McDonald's Q4 2024 Earnings Beat Expectations...",
      "source": "Bloomberg",
      "time_published": "2024-01-30 09:00:00",
      "summary": "McDonald's Corporation reported...",
      "popularity_score": 9.5,
      "url": "https://www.bloomberg.com/news/mcd-q4-earnings",
      "calculated_score": 9.5
    }
  ]
}
```

## Data Sources

### Alpha Vantage
- Provides comprehensive financial news with sentiment analysis
- Free tier: 500 requests/day
- Includes relevance scores and sentiment metrics

### Finnhub
- Offers company-specific news with rich metadata
- Free tier: 60 API calls/minute
- Includes headlines, summaries, and images

### Sample Data
- When API access is unavailable, the script provides representative sample data
- Sample includes typical news categories: earnings, expansion, technology, analyst ratings, etc.

## Popularity Ranking

Articles are ranked using the following criteria (in order of preference):

1. **Explicit Popularity Score**: If provided by the data source
2. **Relevance Score**: From API metadata (scaled 0-10)
3. **Sentiment Score**: Absolute value indicates article impact
4. **Recency Score**: More recent articles ranked higher

## Sample News Categories Covered

The fetcher captures various types of financial news:

- **Earnings Reports**: Quarterly and annual financial results
- **Stock Performance**: Price movements and milestones
- **Business Expansion**: New locations, concepts, and markets
- **Technology Adoption**: AI, digital transformation, automation
- **Analyst Coverage**: Upgrades, downgrades, price targets
- **Dividend Announcements**: Shareholder returns
- **Product Launches**: New menu items and innovations
- **Market Trends**: Industry analysis and competitive positioning

## Troubleshooting

### "No articles found"
- Check your internet connection
- Verify API keys are valid
- Try using sample data mode

### API Rate Limits
- Free tier APIs have request limits
- Wait a few minutes between runs
- Consider upgrading to paid tier for higher limits

### Import Errors
```bash
pip install requests
```

## API Documentation

- **Alpha Vantage**: https://www.alphavantage.co/documentation/
- **Finnhub**: https://finnhub.io/docs/api

## Note on Data Accuracy

- Sample data is representative but not real-time
- For investment decisions, always verify with official sources
- This tool is for informational purposes only

## License

This module is part of the mcmc_gravitationalwave repository.
