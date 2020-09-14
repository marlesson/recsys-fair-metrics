import sys, os
import unittest
from unittest.mock import patch
from recsys_fair_metrics.recsys_fair import RecsysFair

import pandas as pd
import numpy as np
import shutil

OUTPUT_TEST = "tests/output"


class TestRelevanceMetrics(unittest.TestCase):
    def setUp(self):
        # shutil.rmtree(OUTPUT_TEST, ignore_errors=True)
        os.makedirs(OUTPUT_TEST, exist_ok=True)

        self.df = pd.read_csv("tests/factories/test_set_predictions.csv")
        self.recsys_fair = RecsysFair(
            df=self.df[self.df.trained == 0],
            supp_metadata=None,
            user_column="userid",
            item_column="musicbrainz-artist-id",
            reclist_column="sorted_actions",
            reclist_score_column="action_scores",
        )

    def test_metric(self):
        rel = self.recsys_fair.relevance(["precision@1", "ndcg@5", "ndcg@10"])

        metric = rel.metric()
        self.assertEqual(np.round(metric["precision@1"], 4), 0.0596)
        self.assertEqual(np.round(metric["ndcg@5"], 4), 0.1197)

    def test_show(self):
        rel = self.recsys_fair.relevance(["precision@1", "ndcg@5", "ndcg@10"])

        fig = rel.show(title="RankMetrics")
        fig.write_image(OUTPUT_TEST + "/rank_metrics.png")


if __name__ == "__main__":
    unittest.main()
