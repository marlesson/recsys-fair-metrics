import sys, os

import unittest
from unittest.mock import patch
import pandas as pd
from recsys_fair_metrics.recsys_fair import RecsysFair
import shutil

OUTPUT_TEST = "tests/output"


class TestFairnessMistreatment(unittest.TestCase):
    def setUp(self):
        # shutil.rmtree(OUTPUT_TEST, ignore_errors=True)
        os.makedirs(OUTPUT_TEST, exist_ok=True)

        self.df = pd.read_csv("tests/factories/test_set_predictions.csv")
        self.column = "artist_rating"
        self.recsys_fair = RecsysFair(
            df=self.df,
            supp_metadata=None,
            user_column="userid",
            item_column="musicbrainz-artist-id",
            reclist_column="sorted_actions",
            reclist_score_column="action_scores",
        )

    def test_metric(self):
        dm = self.recsys_fair.disparate_mistreatment(self.column)

        metric = dm.metric()
        self.assertEqual(metric["true_positive_rate"].round(4), 0.1128)

    def test_show(self):
        dm = self.recsys_fair.disparate_mistreatment(self.column)

        fig = dm.show()
        fig.write_image(OUTPUT_TEST + "/disparate_mistreatment.png")
