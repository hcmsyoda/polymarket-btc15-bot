"""Tests for market discovery including slug generation, window alignment, and token parsing."""

import unittest
import time
from unittest.mock import MagicMock, patch
from market import (
    current_window_ts,
    next_window_ts,
    market_slug,
    extract_token_ids,
    get_current_market,
    MarketWindow,
    PERIOD_SECONDS,
)


class TestWindowTimestamps(unittest.TestCase):
    def test_current_window_ts(self):
        ts = current_window_ts(5)
        now = int(time.time())
        expected = now - (now % 300)
        self.assertEqual(ts, expected)
        self.assertEqual(ts % 300, 0)

    def test_next_window_ts(self):
        current = current_window_ts(5)
        nxt = next_window_ts(5)
        self.assertEqual(nxt, current + 300)

    def test_market_slug(self):
        slug = market_slug(5, 1700000000)
        self.assertEqual(slug, "btc-updown-5m-1700000000")

    def test_market_slug_default_window(self):
        with patch("market.current_window_ts", return_value=1700000000):
            slug = market_slug(5)
        self.assertEqual(slug, "btc-updown-5m-1700000000")


class TestExtractTokenIds(unittest.TestCase):
    def test_basic(self):
        event = {
            "markets": [{
                "clobTokenIds": '["token_up", "token_down"]',
                "outcomes": '["Up", "Down"]',
            }]
        }
        up, down = extract_token_ids(event)
        self.assertEqual(up, "token_up")
        self.assertEqual(down, "token_down")

    def test_already_parsed(self):
        event = {
            "markets": [{
                "clobTokenIds": ["token_up", "token_down"],
                "outcomes": ["Up", "Down"],
            }]
        }
        up, down = extract_token_ids(event)
        self.assertEqual(up, "token_up")
        self.assertEqual(down, "token_down")

    def test_fallback_order(self):
        event = {
            "markets": [{
                "clobTokenIds": '["abc", "def"]',
                "outcomes": '[]',
            }]
        }
        up, down = extract_token_ids(event)
        self.assertEqual(up, "abc")
        self.assertEqual(down, "def")

    def test_no_markets(self):
        with self.assertRaises(ValueError) as ctx:
            extract_token_ids({"markets": []})
        self.assertIn("No markets", str(ctx.exception))


class TestMarketWindow(unittest.TestCase):
    def test_seconds_remaining(self):
        future = int(time.time()) + 180
        mw = MarketWindow(
            slug="test",
            condition_id="cond1",
            token_id_up="up",
            token_id_down="down",
            window_start=int(time.time()),
            window_end=future,
        )
        self.assertGreater(mw.seconds_remaining, 170)
        self.assertLess(mw.seconds_remaining, 190)


class TestGetCurrentMarket(unittest.TestCase):
    @patch("market.fetch_market_by_slug")
    def test_fetch_failure(self, mock_fetch):
        mock_fetch.return_value = None
        result = get_current_market(5)
        self.assertIsNone(result)

    @patch("market.fetch_market_by_slug")
    def test_basic(self, mock_fetch):
        mock_fetch.return_value = {
            "markets": [{
                "clobTokenIds": '["up1", "down1"]',
                "outcomes": '["Up", "Down"]',
                "conditionId": "cond1",
                "outcomePrices": '["0.6", "0.4"]',
            }]
        }
        result = get_current_market(5)
        self.assertIsNotNone(result)
        self.assertEqual(result.condition_id, "cond1")
        self.assertEqual(result.token_id_up, "up1")
        self.assertEqual(result.token_id_down, "down1")
        self.assertEqual(result.up_price, 0.6)
        self.assertEqual(result.down_price, 0.4)


if __name__ == "__main__":
    unittest.main()
