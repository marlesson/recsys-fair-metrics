import sys, os

import unittest
from unittest.mock import patch
import pandas as pd
from recsys_fair_metrics.recsys_fair import RecsysFair

import shutil
import numpy as np

OUTPUT_TEST = 'tests/output'

class TestExposureMetrics(unittest.TestCase):

    def setUp(self):
      os.makedirs(OUTPUT_TEST, exist_ok=True)

      self.df     = pd.read_csv('tests/factories/test_set_predictions.csv')
      self.recsys_fair = RecsysFair(self.df, 
                                          'userid', 
                                          'musicbrainz-artist-id', 
                                          'sorted_actions', 
                                          'action_scores')
      
    def test_metric(self):
      exp = self.recsys_fair.exposure()

      metric = exp.metric()
      #print(metric)
      #self.assertEqual(np.round(metric['precision@1'], 4), 0.0596)
      #self.assertEqual(np.round(metric['ndcg@5'], 4),0.1197)

    # def test_ndce_at_k(self):
    #   exp = self.recsys_fair.exposure()
    #   supp = 'd43d12a1-2dc9-4257-a2fd-0a3bb1081b86'
    #   metric = exp.ndce_at_k(supp, 20)
    #   self.assertEqual(np.round(metric, 4),0.0414)

    #   supp = 'af1122a8-e2f0-4534-8ce1-919d1edba1d9'
    #   metric = exp.ndce_at_k(supp, 20)
    #   self.assertEqual(np.round(metric, 4),0.0004)

    # def test_show(self):
    #   exp = self.recsys_fair.exposure()

    #   fig = exp.show(title="Exposure")
    #   fig.write_image(OUTPUT_TEST+"/exposure_geral.png")

