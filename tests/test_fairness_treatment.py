import sys, os

import unittest
from unittest.mock import patch
import pandas as pd
from fairness.disparate_treatment import DisparateTreatment
from recsys_fair_metrics import RecsysFairMetrics
import shutil

OUTPUT_TEST = 'tests/output'

class TestFairnessTreatment(unittest.TestCase):

    def setUp(self):
      #shutil.rmtree(OUTPUT_TEST, ignore_errors=True)
      os.makedirs(OUTPUT_TEST, exist_ok=True)

      self.df     = pd.read_csv('tests/factories/test_set_predictions.csv')
      self.column = "user_gender"
      self.recsys_fair = RecsysFairMetrics(self.df, 
                                          'userid', 
                                          'musicbrainz-artist-id', 
                                          'sorted_actions', 
                                          'action_scores')
      
    def test_metric(self):
      dt = self.recsys_fair.disparate_treatment(self.column)

      metric = dt.metric()
      self.assertEqual(metric.round(4), 0.5054)


    def test_show_bar(self):
      dt = self.recsys_fair.disparate_treatment(self.column)

      fig = dt.show(kind='bar', title="Disparate Treatment", top_k = 9)
      fig.write_image(OUTPUT_TEST+"/disparate_treatment_bar.png")

    def test_show_ks(self):
      dt = self.recsys_fair.disparate_treatment(self.column)

      fig = dt.show(kind='ks', title="Disparate Treatment")
      fig.write_image(OUTPUT_TEST+"/disparate_treatment_ks.png")      