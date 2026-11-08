import unittest
from eval.evaluate import _mean, _token_total, aggregate

class TestEvaluation(unittest.TestCase):
    def test_helper_calculations(self):
        usage = {
            "input_tokens": 400,
            "output_tokens": 100,
        }
        self.assertEqual(_token_total(usage),500)
        self.assertEqual(_mean([1, 2, 3]),2.0)

    def test_aggregate_metrics(self):
        rows = [
            {
                "code_ok": True,
                "resolved": True,
                "action_ok": True,
                "evidence_valid": True,
                "evidence_relevant": True,
                "hallucinated_evidence": False,
                "escalated": False,
                "latency_seconds": 1.0,
                "total_tokens": 100,
            },
            {
                "code_ok": False,
                "resolved": False,
                "action_ok": False,
                "evidence_valid": False,
                "evidence_relevant": False,
                "hallucinated_evidence": True,
                "escalated": True,
                "latency_seconds": 3.0,
                "total_tokens": 300,
            },
        ]
        summary = aggregate(rows)
        self.assertEqual(summary["total_cases"],2)
        self.assertEqual(summary["root_cause_accuracy_pct"],50.0)
        self.assertEqual(summary["action_accuracy_pct"],50.0)
        self.assertEqual(summary["evidence_valid_pct"],50.0)
        self.assertEqual(summary["hallucinated_evidence_pct"],50.0)
        self.assertEqual(summary["avg_latency_seconds"],2.0)
        self.assertEqual(summary["avg_total_tokens"],200.0)

    def test_empty_results(self):
        self.assertEqual(aggregate([]),{})
if __name__ == "__main__":
    unittest.main()