"""Tests for CLOB v2 executor including initialization, order sizing, and fee logic."""

import unittest
from unittest.mock import MagicMock, patch
from executor import (
    Executor,
    MarketMeta,
    BuilderConfig,
    FILLED,
    FAILED,
    MAX_BUY_PRICE,
)


class TestExecutorDryRun(unittest.TestCase):
    def setUp(self):
        self.ex = Executor(private_key="fake_key", safe_address="", dry_run=True)

    def test_initialize_dry_run(self):
        with patch("executor.ClobClient") as MockClient:
            mock_client = MagicMock()
            mock_client.derive_api_key.return_value = {"apiKey": "test_key", "secret": "test_secret", "passphrase": "test_pass"}
            MockClient.return_value = mock_client
            self.assertTrue(self.ex.initialize())
            self.assertTrue(self.ex._initialized)

    def test_get_collateral_balance_dry_run(self):
        with patch("executor.ClobClient") as MockClient:
            mock_client = MagicMock()
            mock_client.derive_api_key.return_value = {"apiKey": "test_key", "secret": "test_secret", "passphrase": "test_pass"}
            mock_client.get_balance_allowance.return_value = {"balance": "100000000"}  # $100 in 6-decimal units
            MockClient.return_value = mock_client
            self.ex.initialize()
            self.assertEqual(self.ex.get_collateral_balance(), 100.0)

    def test_get_fee_rate_bps_dry_run(self):
        with patch("executor.ClobClient") as MockClient:
            mock_client = MagicMock()
            mock_client.derive_api_key.return_value = {"apiKey": "test_key", "secret": "test_secret", "passphrase": "test_pass"}
            MockClient.return_value = mock_client
            self.ex.initialize()
            self.assertEqual(self.ex.get_fee_rate_bps("cond1"), 0)

    def test_buy_dry_run(self):
        with patch("executor.ClobClient") as MockClient:
            mock_client = MagicMock()
            mock_client.derive_api_key.return_value = {"apiKey": "test_key", "secret": "test_secret", "passphrase": "test_pass"}
            MockClient.return_value = mock_client
            self.ex.initialize()
        result = self.ex.buy("token1", 10.0, 0.60)
        self.assertEqual(result["status"], FILLED)
        self.assertTrue(result["dry_run"])
        self.assertAlmostEqual(result["shares"], 16.67, places=1)

    def test_buy_price_too_high(self):
        self.ex.initialize()
        result = self.ex.buy("token1", 10.0, 0.99)
        self.assertEqual(result["status"], FAILED)

    def test_sell_dry_run(self):
        self.ex.initialize()
        result = self.ex.sell("token1", 10.0, 0.70)
        self.assertEqual(result["status"], FILLED)
        self.assertTrue(result["dry_run"])

    def test_sell_below_minimum(self):
        self.ex.initialize()
        result = self.ex.sell("token1", 1.0, 0.50)
        self.assertEqual(result["status"], FAILED)

    def test_calculate_order_size(self):
        shares, cost = self.ex.calculate_order_size(10.0, 0.50, fee_rate_bps=100)
        self.assertEqual(shares, 20.0)
        self.assertAlmostEqual(cost, 10.10, places=2)

    def test_apply_fee(self):
        self.assertEqual(self.ex.apply_fee_to_cost(100.0, 200), 102.0)
        self.assertEqual(self.ex.apply_fee_to_payout(100.0, 200), 98.0)

    def test_cancel_all_dry_run(self):
        result = self.ex.cancel_all_orders()
        self.assertTrue(result["dry_run"])

    def test_builder_config_empty(self):
        self.assertIsNone(self.ex.builder_config.to_dict())


class TestExecutorLiveMocked(unittest.TestCase):
    def setUp(self):
        self.ex = Executor(private_key="0x" + "a" * 64, safe_address="", dry_run=False)

    @patch("executor.ClobClient")
    def test_initialize_without_safe(self, mock_client_cls):
        mock_client = MagicMock()
        mock_creds = MagicMock()
        mock_client.create_or_derive_api_creds.return_value = mock_creds
        mock_client_cls.return_value = mock_client

        result = self.ex.initialize()
        self.assertTrue(result)
        mock_client_cls.assert_called_once()
        call_kwargs = mock_client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["chain_id"], 137)
        self.assertNotIn("funder", call_kwargs)
        self.assertNotIn("signature_type", call_kwargs)
        mock_client.set_api_creds.assert_called_once_with(mock_creds)

    @patch("executor.ClobClient")
    def test_initialize_with_safe(self, mock_client_cls):
        ex = Executor(private_key="0x" + "a" * 64, safe_address="0xSafeAddr", dry_run=False)
        mock_client = MagicMock()
        mock_creds = MagicMock()
        mock_client.create_or_derive_api_creds.return_value = mock_creds
        mock_client_cls.return_value = mock_client

        result = ex.initialize()
        self.assertTrue(result)
        call_kwargs = mock_client_cls.call_args.kwargs
        self.assertEqual(call_kwargs["funder"], "0xSafeAddr")
        self.assertEqual(call_kwargs["signature_type"], 2)

    @patch("executor.ClobClient")
    def test_get_collateral_balance(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_balance_allowance.return_value = {"balance": "5000000"}
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        balance = self.ex.get_collateral_balance()
        self.assertEqual(balance, 5.0)
        mock_client.get_balance_allowance.assert_called_once()
        call_args = mock_client.get_balance_allowance.call_args[0][0]
        self.assertEqual(call_args.asset_type, "COLLATERAL")

    @patch("executor.ClobClient")
    def test_buy_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.create_order.return_value = {"success": True, "orderID": "ord123"}
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        result = self.ex.buy("token1", 12.0, 0.60)
        self.assertEqual(result["status"], FILLED)
        self.assertEqual(result["order_id"], "ord123")
        mock_client.create_order.assert_called_once()

    @patch("executor.ClobClient")
    def test_buy_rejected_price_too_high(self, mock_client_cls):
        self.ex.client = MagicMock()
        self.ex._initialized = True
        result = self.ex.buy("token1", 10.0, 0.99)
        self.assertEqual(result["status"], FAILED)

    @patch("executor.ClobClient")
    def test_sell_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.create_order.return_value = {"success": True, "orderID": "ord456"}
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        result = self.ex.sell("token1", 10.0, 0.70)
        self.assertEqual(result["status"], FILLED)
        self.assertEqual(result["order_id"], "ord456")

    @patch("executor.ClobClient")
    def test_sell_below_minimum_notional(self, mock_client_cls):
        self.ex.client = MagicMock()
        self.ex._initialized = True
        result = self.ex.sell("token1", 1.0, 0.50)
        self.assertEqual(result["status"], FAILED)

    @patch("executor.ClobClient")
    def test_get_market_meta(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "active": True,
            "closed": False,
            "fee_rate_bps": "100",
            "min_order_size": "5",
            "min_tick_size": "0.01",
            "tokens": ["t1", "t2"],
        }
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        meta = self.ex.refresh_market_meta("cond1")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.fee_rate_bps, 100)
        self.assertTrue(meta.active)

    @patch("executor.ClobClient")
    def test_get_fee_rate_bps(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "active": True,
            "closed": False,
            "fee_rate_bps": "150",
        }
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        fee = self.ex.get_fee_rate_bps("cond1")
        self.assertEqual(fee, 150)

    @patch("executor.ClobClient")
    def test_fee_cache(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "active": True,
            "closed": False,
            "fee_rate_bps": "200",
        }
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        fee1 = self.ex.get_fee_rate_bps("cond1")
        fee2 = self.ex.get_fee_rate_bps("cond1")
        self.assertEqual(fee1, 200)
        self.assertEqual(fee2, 200)
        # Should hit cache second time
        self.assertEqual(mock_client.get_market.call_count, 1)

    @patch("executor.ClobClient")
    def test_is_market_tradable(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "active": True,
            "closed": False,
            "fee_rate_bps": "0",
        }
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        self.assertTrue(self.ex.is_market_tradable("cond1"))

    @patch("executor.ClobClient")
    def test_market_meta_cache(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "active": True,
            "closed": False,
            "fee_rate_bps": "50",
        }
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        self.ex.refresh_market_meta("cond1")
        self.assertIn("cond1", self.ex._market_meta_cache)
        self.assertTrue(self.ex.is_market_tradable("cond1"))

    @patch("executor.ClobClient")
    def test_cancel_all(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.cancel_all.return_value = ["ord1", "ord2"]
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        result = self.ex.cancel_all_orders()
        self.assertEqual(result["cancelled"], 2)

    @patch("executor.ClobClient")
    def test_cancel_order(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        self.assertTrue(self.ex.cancel_order("ord1"))
        mock_client.cancel.assert_called_once_with(order_id="ord1")

    @patch("executor.ClobClient")
    def test_buy_market_not_tradable(self, mock_client_cls):
        # Note: the current executor.buy does not check is_market_tradable,
        # but we verify the helper works for callers that do use it.
        mock_client = MagicMock()
        mock_client.get_market.return_value = {
            "active": False,
            "closed": True,
            "fee_rate_bps": "0",
        }
        mock_client_cls.return_value = mock_client

        self.ex.client = mock_client
        self.ex._initialized = True
        self.assertFalse(self.ex.is_market_tradable("cond1"))


class TestBuilderConfig(unittest.TestCase):
    def test_load_builder_config(self):
        with patch.dict("os.environ", {"BUILDER_ADDRESS": "0xBuilder", "BUILDER_FEE": "100"}):
            ex = Executor()
            d = ex.builder_config.to_dict()
            self.assertIsNotNone(d)
            self.assertEqual(d["builder"], "0xBuilder")
            self.assertEqual(d["fee"], "100")

    def test_load_builder_config_empty(self):
        with patch.dict("os.environ", {}, clear=True):
            ex = Executor()
            self.assertIsNone(ex.builder_config.to_dict())


if __name__ == "__main__":
    unittest.main()
