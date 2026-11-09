import sys, unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from risk import RiskManager  # noqa: E402


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(target_pct=10)
        self.rm.new_day(date(2026, 1, 5), 10_000)

    def test_lot_size_risk_05pct(self):
        # 50 EUR risque, stop 5$, eurusd 1.0 -> 50 / (5*100) = 0.10 lot
        self.assertAlmostEqual(self.rm.lot_size(5.0, 1.0, 10_000), 0.10)

    def test_lot_size_zero_if_below_min(self):
        self.assertEqual(self.rm.lot_size(500.0, 1.0, 10_000), 0.0)

    def test_daily_stop(self):
        self.assertIsNotNone(self.rm.check(10_000, 9_800))
        self.assertIsNone(self.rm.check(10_000, 9_850))

    def test_lot_reduced_near_daily_stop(self):
        # 180 perdus sur 200 autorisés -> 20 EUR de marge -> 0.04 lot max
        self.assertAlmostEqual(self.rm.lot_size(5.0, 1.0, 9_820), 0.04)

    def test_max_loss_halts(self):
        self.rm.new_day(date(2026, 1, 6), 9_450)
        self.assertTrue(self.rm.check(9_450, 9_400).startswith("STOP"))
        self.assertIsNotNone(self.rm.can_open(9_500, 9_500, 0.2, False))  # définitif

    def test_target_halts(self):
        self.assertTrue(self.rm.check(11_000, 11_000).startswith("OBJECTIF"))

    def test_limits(self):
        self.assertEqual(self.rm.can_open(10_000, 10_000, 0.9, False), "spread trop large")
        self.rm.on_close(-50); self.rm.on_close(-50)
        self.assertEqual(self.rm.can_open(10_000, 10_000, 0.2, False), "pertes consecutives du jour")


if __name__ == "__main__":
    unittest.main()
