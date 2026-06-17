"""Tests for strategy engine including probability, Kelly sizing, and fee-aware rejection."""

import unittest
import math
from strategy import (
    estimate_true_probability,
    kelly_bet_size,
    evaluate,
    get_skip_reason,
    StrategyConfig,
    TradeSignal,
    TradingStats,
    HourlyStats,
)


class TestEstimateTrueProbability(unittest.TestCase):
    def test_zero_delta(self):
        prob = estimate_true_probability(0.0, 150, vol=0.12)
        self.assertAlmostEqual(prob, 0.5, places=2)

    def test_large_delta(self):
        prob = estimate_true_probability(0.5, 150, vol=0.12)
        self.assertGreater(prob, 0.95)

    def test_negative_delta(self):
        prob = estimate_true_probability(-0.3, 150, vol=0.12)
        self.assertGreater(prob, 0.5)
        self.assertLess(prob, 1.0)

    def test_clamped(self):
        prob = estimate_true_probability(10.0, 1, vol=0.01)
        self.assertEqual(prob, 0.99)

    def test_time_factor(self):
        # Less time remaining -> higher effective vol -> higher prob for same delta
        prob1 = estimate_true_probability(0.2, 300, vol=0.12)
        prob2 = estimate_true_probability(0.2, 1, vol=0.12)
        self.assertGreater(prob2, prob1)


class TestKellyBetSize(unittest.TestCase):
    def test_basic(self):
        # p=0.9, price=0.6, b=(1-0.6)/0.6=0.6667, kelly=(0.6667*0.9-0.1)/0.6667=0.75
        # quarter kelly -> 0.1875 * 100 = 18.75
        bet = kelly_bet_size(0.9, 0.6, 100.0, fraction=0.25, min_bet=5.0, max_bet=25.0)
        self.assertAlmostEqual(bet, 18.75, places=2)

    def test_min_bet(self):
        bet = kelly_bet_size(0.9, 0.6, 100.0, fraction=0.01, min_bet=5.0, max_bet=25.0)
        self.assertEqual(bet, 5.0)

    def test_max_bet(self):
        bet = kelly_bet_size(0.99, 0.51, 10000.0, fraction=1.0, min_bet=1.0, max_bet=25.0)
        self.assertEqual(bet, 25.0)

    def test_zero_price(self):
        bet = kelly_bet_size(0.9, 0.0, 100.0)
        self.assertEqual(bet, 0.0)

    def test_negative_kelly(self):
        # p < price means negative expected value
        bet = kelly_bet_size(0.5, 0.6, 100.0)
        self.assertEqual(bet, 0.0)

    def test_fee_reduces_size(self):
        bet_no_fee = kelly_bet_size(0.9, 0.6, 100.0, fraction=0.25)
        bet_with_fee = kelly_bet_size(0.9, 0.6, 100.0, fraction=0.25, fee_rate_bps=200)
        self.assertLess(bet_with_fee, bet_no_fee)

    def test_fee_makes_unprofitable(self):
        # p=0.6, price=0.6, 100bps fee -> effective_price=0.606, negative kelly
        bet = kelly_bet_size(0.6, 0.6, 100.0, fee_rate_bps=100)
        self.assertEqual(bet, 0.0)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig(
            min_edge=0.05,
            min_prob=0.80,
            entry_window_start=240,
            entry_window_end=10,
            min_btc_delta=0.06,
            kelly_fraction=0.25,
            min_bet=5.0,
            max_bet=25.0,
        )

    def test_entry_allowed(self):
        # BTC up 0.07% with 150s left -> should be strong signal
        signal = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.side, "UP")
        self.assertGreater(signal.edge, 0.05)

    def test_too_early(self):
        signal = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=250,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNone(signal)

    def test_too_late(self):
        signal = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=5,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNone(signal)

    def test_delta_too_small(self):
        signal = evaluate(
            btc_price=70003.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNone(signal)

    def test_price_out_of_range(self):
        signal = evaluate(
            btc_price=72000.0,
            opening_price=70000.0,
            up_market_price=0.95,
            down_market_price=0.05,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNone(signal)

    def test_prob_below_min(self):
        signal = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=290,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNone(signal)

    def test_edge_below_min(self):
        # true_prob ~0.80, price=0.78 -> edge=0.02 < 0.05
        signal = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.78,
            down_market_price=0.22,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
        )
        self.assertIsNone(signal)

    def test_fee_aware_rejection(self):
        """Fee should eat a razor-thin edge and reject the trade."""
        # No fee: edge is ~0.050 (just above threshold)
        signal_no_fee = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.75,
            down_market_price=0.25,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
            fee_rate_bps=0,
        )
        self.assertIsNotNone(signal_no_fee)

        # 200 bps fee on 0.75 price = 0.015 cost, edge drops to ~0.035
        signal_with_fee = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.75,
            down_market_price=0.25,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
            fee_rate_bps=200,
        )
        self.assertIsNone(signal_with_fee)

    def test_fee_reduces_kelly(self):
        """Same valid trade with fee should result in smaller Kelly size."""
        signal_no_fee = evaluate(
            btc_price=70100.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
            fee_rate_bps=0,
        )
        signal_low_fee = evaluate(
            btc_price=70100.0,
            opening_price=70000.0,
            up_market_price=0.60,
            down_market_price=0.40,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
            fee_rate_bps=50,  # 0.5%
        )
        self.assertIsNotNone(signal_no_fee)
        self.assertIsNotNone(signal_low_fee)
        self.assertLess(signal_low_fee.kelly_size, signal_no_fee.kelly_size)

    def test_realized_vol(self):
        # Higher vol should lower probability for same delta
        signal_low_vol = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.55,
            down_market_price=0.45,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
            realized_vol=0.06,
        )
        signal_high_vol = evaluate(
            btc_price=70050.0,
            opening_price=70000.0,
            up_market_price=0.55,
            down_market_price=0.45,
            seconds_remaining=150,
            bankroll=100.0,
            config=self.config,
            realized_vol=0.12,
        )
        self.assertIsNotNone(signal_low_vol)
        self.assertIsNotNone(signal_high_vol)
        self.assertGreater(signal_low_vol.true_prob, signal_high_vol.true_prob)


class TestGetSkipReason(unittest.TestCase):
    def setUp(self):
        self.config = StrategyConfig()

    def test_delta_too_small(self):
        reason = get_skip_reason(
            70001.0, 70000.0, 0.60, 0.40, 150, config=self.config
        )
        self.assertEqual(reason, "delta_too_small")

    def test_price_out_of_range(self):
        reason = get_skip_reason(
            72000.0, 70000.0, 0.95, 0.05, 150, config=self.config
        )
        self.assertEqual(reason, "price_out_of_range")

    def test_prob_below_min(self):
        # small delta above min but still low prob
        reason = get_skip_reason(
            70050.0, 70000.0, 0.60, 0.40, 290, config=self.config
        )
        self.assertEqual(reason, "prob_below_min")

    def test_no_skip(self):
        reason = get_skip_reason(
            72000.0, 70000.0, 0.60, 0.40, 150, config=self.config
        )
        self.assertEqual(reason, "")


class TestTradingStats(unittest.TestCase):
    def test_record_win(self):
        stats = TradingStats(bankroll=100.0)
        stats.record_win(5.0)
        self.assertEqual(stats.total_trades, 1)
        self.assertEqual(stats.wins, 1)
        self.assertEqual(stats.bankroll, 105.0)

    def test_record_loss(self):
        stats = TradingStats(bankroll=100.0)
        stats.record_loss(3.0)
        self.assertEqual(stats.total_trades, 1)
        self.assertEqual(stats.losses, 1)
        self.assertEqual(stats.bankroll, 97.0)

    def test_hourly_reset(self):
        hourly = HourlyStats()
        hourly.record_trade(0.1, 0.2)
        hourly.reset()
        self.assertEqual(hourly.trades, 0)
        self.assertEqual(len(hourly.edges), 0)


if __name__ == "__main__":
    unittest.main()
