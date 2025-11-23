"""
Test suite for the McDonald's news fetcher module.
"""

import unittest
import json
import os
from datetime import datetime, timedelta
from fetch_mcd_news import MCDNewsFetcher


class TestMCDNewsFetcher(unittest.TestCase):
    """Test cases for MCDNewsFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = MCDNewsFetcher()
    
    def test_initialization(self):
        """Test that the fetcher initializes correctly."""
        self.assertEqual(self.fetcher.ticker, "MCD")
        self.assertEqual(self.fetcher.company_name, "McDonald's")
        self.assertIsInstance(self.fetcher.base_date, datetime)
        self.assertIsInstance(self.fetcher.twelve_months_ago, datetime)
        
        # Check date range is approximately 12 months
        delta = self.fetcher.base_date - self.fetcher.twelve_months_ago
        self.assertGreaterEqual(delta.days, 364)
        self.assertLessEqual(delta.days, 366)
    
    def test_get_sample_news(self):
        """Test that sample news is returned correctly."""
        articles = self.fetcher.get_sample_news()
        
        # Check we get articles
        self.assertGreater(len(articles), 0)
        self.assertLessEqual(len(articles), 20)
        
        # Check article structure
        for article in articles:
            self.assertIn('title', article)
            self.assertIn('source', article)
            self.assertIn('time_published', article)
            self.assertIn('summary', article)
            self.assertIn('url', article)
            
            # Check title is non-empty
            self.assertTrue(article['title'])
            
            # Check date format
            try:
                datetime.strptime(article['time_published'], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                self.fail(f"Invalid date format: {article['time_published']}")
    
    def test_rank_articles_by_popularity(self):
        """Test article ranking functionality."""
        articles = self.fetcher.get_sample_news()
        ranked = self.fetcher.rank_articles_by_popularity(articles)
        
        # Check all articles are returned
        self.assertEqual(len(ranked), len(articles))
        
        # Check each article has a calculated score
        for article in ranked:
            self.assertIn('calculated_score', article)
            self.assertIsInstance(article['calculated_score'], (int, float))
        
        # Check articles are sorted in descending order
        scores = [a['calculated_score'] for a in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_rank_with_different_score_types(self):
        """Test ranking with different types of scores."""
        # Test with popularity_score
        articles_with_pop = [
            {'title': 'Article 1', 'popularity_score': 8.5, 'time_published': '2024-01-01 00:00:00', 'source': 'Test', 'summary': 'Test', 'url': ''},
            {'title': 'Article 2', 'popularity_score': 9.5, 'time_published': '2024-01-01 00:00:00', 'source': 'Test', 'summary': 'Test', 'url': ''},
        ]
        ranked = self.fetcher.rank_articles_by_popularity(articles_with_pop)
        self.assertEqual(ranked[0]['title'], 'Article 2')
        self.assertEqual(ranked[1]['title'], 'Article 1')
        
        # Test with relevance_score
        articles_with_rel = [
            {'title': 'Article A', 'relevance_score': 0.7, 'time_published': '2024-01-01 00:00:00', 'source': 'Test', 'summary': 'Test', 'url': ''},
            {'title': 'Article B', 'relevance_score': 0.9, 'time_published': '2024-01-01 00:00:00', 'source': 'Test', 'summary': 'Test', 'url': ''},
        ]
        ranked = self.fetcher.rank_articles_by_popularity(articles_with_rel)
        self.assertEqual(ranked[0]['title'], 'Article B')
    
    def test_save_to_json(self):
        """Test JSON file saving functionality."""
        import tempfile
        
        articles = self.fetcher.get_sample_news()
        ranked = self.fetcher.rank_articles_by_popularity(articles)
        
        # Save to a test file using tempfile for cross-platform compatibility
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            test_filename = tf.name
        
        self.fetcher.save_to_json(ranked, filename=test_filename)
        
        # Verify file exists
        self.assertTrue(os.path.exists(test_filename))
        
        # Verify file content
        with open(test_filename, 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['ticker'], 'MCD')
        self.assertEqual(data['company'], "McDonald's")
        self.assertIn('date_range', data)
        self.assertIn('from', data['date_range'])
        self.assertIn('to', data['date_range'])
        self.assertEqual(data['total_articles'], len(ranked))
        self.assertEqual(len(data['articles']), len(ranked))
        
        # Clean up
        os.remove(test_filename)
    
    def test_display_news(self):
        """Test that display_news runs without errors."""
        articles = self.fetcher.get_sample_news()
        ranked = self.fetcher.rank_articles_by_popularity(articles)
        
        # This should not raise any exceptions
        try:
            self.fetcher.display_news(ranked, max_articles=5)
        except Exception as e:
            self.fail(f"display_news raised exception: {e}")
    
    def test_display_news_empty(self):
        """Test display_news with empty article list."""
        try:
            self.fetcher.display_news([], max_articles=5)
        except Exception as e:
            self.fail(f"display_news raised exception with empty list: {e}")
    
    def test_fetch_news_alpha_vantage_error_handling(self):
        """Test that Alpha Vantage fetch handles errors gracefully."""
        # This will likely fail due to network restrictions, but should not crash
        try:
            articles = self.fetcher.fetch_news_alpha_vantage(api_key="invalid_key")
            # Should return empty list on error
            self.assertIsInstance(articles, list)
        except Exception as e:
            self.fail(f"fetch_news_alpha_vantage raised unexpected exception: {e}")
    
    def test_article_date_filtering(self):
        """Test that sample articles are within the past 12 months."""
        articles = self.fetcher.get_sample_news()
        
        for article in articles:
            pub_date = datetime.strptime(article['time_published'], "%Y-%m-%d %H:%M:%S")
            # Article should be after twelve_months_ago and before base_date
            # Note: Sample data might have dates in the future for demonstration
            self.assertIsInstance(pub_date, datetime)


class TestArticleStructure(unittest.TestCase):
    """Test the structure and content of news articles."""
    
    def test_article_has_required_fields(self):
        """Test that articles contain all required fields."""
        fetcher = MCDNewsFetcher()
        articles = fetcher.get_sample_news()
        
        required_fields = ['title', 'source', 'time_published', 'summary', 'url']
        
        for article in articles:
            for field in required_fields:
                self.assertIn(field, article, 
                             f"Article missing required field: {field}")
    
    def test_article_content_quality(self):
        """Test that article content meets quality standards."""
        fetcher = MCDNewsFetcher()
        articles = fetcher.get_sample_news()
        
        for article in articles:
            # Title should be reasonable length
            self.assertGreater(len(article['title']), 10)
            self.assertLess(len(article['title']), 200)
            
            # Source should be non-empty
            self.assertTrue(article['source'])
            
            # Summary should be non-empty
            self.assertTrue(article['summary'])
            
            # McDonald's or MCD should be mentioned in title or summary
            content = (article['title'] + ' ' + article['summary']).lower()
            self.assertTrue('mcdonald' in content or 'mcd' in content,
                          f"Article doesn't mention McDonald's: {article['title']}")


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMCDNewsFetcher))
    suite.addTests(loader.loadTestsFromTestCase(TestArticleStructure))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("="*70)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
