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
      self.supp_metadata = pd.read_csv('tests/factories/artist-metadata.csv')
      self.column = "artist_rating"
      self.recsys_fair = RecsysFair(df = self.df, 
                                    supp_metadata = self.supp_metadata,
                                    user_column = 'userid', 
                                    item_column = 'musicbrainz-artist-id', 
                                    reclist_column = 'sorted_actions', 
                                    reclist_score_column = 'action_scores')
      
    def test_metric(self):
      exp = self.recsys_fair.exposure(self.column, k=10)

      metric = exp.metric()

      print(metric)

    def test_ndce_at_k(self):
      exp = self.recsys_fair.exposure(self.column)
      supp = 'd43d12a1-2dc9-4257-a2fd-0a3bb1081b86'
      metric = exp.ndce_at_k(supp)
      self.assertEqual(np.round(metric, 4),0.0413)

      supp = 'af1122a8-e2f0-4534-8ce1-919d1edba1d9'
      metric = exp.ndce_at_k(supp)
      self.assertEqual(np.round(metric, 4),0.0)

    def test_prob_exp(self):
      exp = self.recsys_fair.exposure(self.column)
      prop = 10000
      supp = ['d43d12a1-2dc9-4257-a2fd-0a3bb1081b86']
      metric = exp.prob_exp(supp, prop)
      self.assertEqual(np.round(metric, 2),351.46)

      supp = ['af1122a8-e2f0-4534-8ce1-919d1edba1d9']
      metric = exp.prob_exp(supp, prop)
      self.assertEqual(np.round(metric, 2),0.29)

      # supp = ['d43d12a1-2dc9-4257-a2fd-0a3bb1081b86', 'af1122a8-e2f0-4534-8ce1-919d1edba1d9']
      # metric = exp.prob_exp(supp, prop)
      # self.assertEqual(np.round(metric, 2), 175.88)

    def test_show(self):
      exp = self.recsys_fair.exposure(self.column)

      fig = exp.show('geral', title="Exposure")
      fig.write_image(OUTPUT_TEST+"/exposure_geral.png")

    def test_show_per_group(self):
      exp = self.recsys_fair.exposure(self.column, k=10)

      fig = exp.show('per_group', prop=100, column=self.column, title="Exposure per Group")
      fig.write_image(OUTPUT_TEST+"/exposure_geral_per_group.png")
