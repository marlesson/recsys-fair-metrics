import sys, os

import unittest
from unittest.mock import patch
import pandas as pd
from recsys_fair_metrics import RecsysFairMetrics
import shutil
import numpy as np

OUTPUT_TEST = 'tests/output'

class TestRelevanceMetrics(unittest.TestCase):

    def setUp(self):
      #shutil.rmtree(OUTPUT_TEST, ignore_errors=True)
      os.makedirs(OUTPUT_TEST, exist_ok=True)

      self.df     = pd.read_csv('tests/factories/test_set_predictions.csv')
      self.recsys_fair = RecsysFairMetrics(self.df[self.df.trained == 0], 
                                          'userid', 
                                          'musicbrainz-artist-id', 
                                          'sorted_actions', 
                                          'action_scores')
      
    def test_metric(self):
      rel = self.recsys_fair.relevance(['precision@1', 'ndcg@5', 'ndcg@10'])

      metric = rel.metric()
      self.assertEqual(np.round(metric['precision@1'], 4), 0.0596)
      self.assertEqual(np.round(metric['ndcg@5'], 4),0.1197)


    def test_show(self):
      rel = self.recsys_fair.relevance(['precision@1', 'ndcg@5', 'ndcg@10'])

      fig = rel.show(title="RankMetrics")
      fig.write_image(OUTPUT_TEST+"/rank_metrics.png")

