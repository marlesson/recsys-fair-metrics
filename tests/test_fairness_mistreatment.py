import sys, os

import unittest
from unittest.mock import patch
import pandas as pd
from fairness.disparate_mistreatment import DisparateMistreatment
from recsys_fair_metrics import RecsysFairMetrics
import shutil

OUTPUT_TEST = 'tests/output'

class TestFairnessMistreatment(unittest.TestCase):

    def setUp(self):
      #shutil.rmtree(OUTPUT_TEST, ignore_errors=True)
      os.makedirs(OUTPUT_TEST, exist_ok=True)

      self.df     = pd.read_csv('tests/factories/test_set_predictions.csv')
      self.column = "artist_rating"
      self.recsys_fair = RecsysFairMetrics(self.df, 
                                          'userid', 
                                          'musicbrainz-artist-id', 
                                          'sorted_actions', 
                                          'action_scores')
      
    def test_metric(self):
      dm = self.recsys_fair.disparate_mistreatment(self.column)

      metric = dm.metric()
      self.assertEqual(metric.round(4), 0.0399)


    def test_show(self):
      dm = self.recsys_fair.disparate_mistreatment(self.column)

      fig = dm.show()
      fig.write_image(OUTPUT_TEST+"/disparate_mistreatment.png")