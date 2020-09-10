import pandas as pd
import numpy as np
from typing import List, Dict, Any
from util.util import mean_confidence_interval
import os
from tqdm import tqdm
from multiprocessing.pool import Pool
from util.rank_metrics import ndcg_at_k, precision_at_k
import functools
from util.util import mean_confidence_interval
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
TEMPLATE = "plotly_white" 

def _create_relevance_list(
    sorted_actions: List[Any], expected_action: Any
) -> List[int]:
    return [1 if str(action) == str(expected_action) else 0 for action in sorted_actions]

def _color_by_metric(metric):
    if "ndcg" in metric:
        return "#C44E51"
    elif "coverage" in metric:
        return "#DD8452"
    elif "personalization" in metric:
        return "#55A868"
    elif "count" in metric:
        return "#CCB974"
    else:
        return "#8C8C8C"


class RelevanceRank(object):
  def __init__(
      self,
      dataframe: pd.DataFrame,
      metrics: List[str],
      item_column: str,
      rec_list: str
  ):
    self._dataframe = dataframe.fillna("-")
    self._item_column = item_column
    self._rec_list = rec_list
    self._metrics = metrics
    self._func_metric = dict(ndcg = ndcg_at_k, precision = precision_at_k)
    self._df_metrics = None
    self.fit(self._dataframe, self._item_column, self._rec_list)

  def fit(
      self,
      df: pd.DataFrame, 
      item_column: str, 
      rec_list: str
  ) -> pd.DataFrame:

    df = df[[item_column, rec_list]]

    with Pool(os.cpu_count()) as p:
      print("Creating the relevance lists...")
      # from IPython import embed; embed()
      df["relevance_list"] = list(
          tqdm(
              p.starmap(
                  _create_relevance_list,
                  zip(
                      df[rec_list],
                      df[item_column]
                  ),
              ),
              total=len(df),
          )
      )

      for m in self._metrics:
        metric, k = m.split("@")

        print("Calculating {} at {}...".format(metric, k))
        df[m] = list(
            tqdm(
                p.map(functools.partial(self._func_metric[metric], k=int(k)), df["relevance_list"]),
                total=len(df),
            )
        )

    self._df_metrics = df


  def metric(self):
    '''
    Rank Metrics - 
    '''
    df = self._df_metrics[self._metrics].mean().to_dict()
    return df


  def show(self, **kwargs):
    
    title  = "Rank Metrics"  if 'title' not in kwargs else kwargs['title']
    data   = []

    #for metric, val in self.metric.items():
    data.append(
        go.Bar(
            #name=list(self.metric.keys()),
            x=list(self.metric().keys()),
            y=list(self.metric().values()),
            marker_color=[_color_by_metric(m) for m in self.metric().keys()],
            error_y=dict(type="data", array = [mean_confidence_interval(self._df_metrics[m])[1] for m in self.metric().keys()]),
            orientation="v",
        )
    )  # px.colors.sequential.Purp [i for i in range(len(rows))]

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        xaxis_title="Metrics",
        legend=dict(y=-0.2),
        title=title,
    )

    return fig