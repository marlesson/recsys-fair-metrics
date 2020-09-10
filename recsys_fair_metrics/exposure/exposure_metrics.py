import pandas as pd
import numpy as np
from typing import List, Dict, Any

import os
from tqdm import tqdm
from multiprocessing.pool import Pool
import functools

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

from recsys_fair_metrics.util.util import mean_confidence_interval
from recsys_fair_metrics.util.rank_metrics import ndcg_at_k
from recsys_fair_metrics.util.rank_metrics import ndcg_at_k, precision_at_k
TEMPLATE = "plotly_white" 

def _create_relevance_list(
    sorted_actions: List[Any], expected_action: Any
) -> List[int]:
    return [1 if str(action) == str(expected_action) else 0 for action in sorted_actions]

# def _color_by_metric(metric):
#     if "ndcg" in metric:
#         return "#C44E51"
#     elif "coverage" in metric:
#         return "#DD8452"
#     elif "personalization" in metric:
#         return "#55A868"
#     elif "count" in metric:
#         return "#CCB974"
#     else:
#         return "#8C8C8C"

class ExposureMetric(object):
  def __init__(
      self,
      dataframe: pd.DataFrame,
      user_column: str,
      rec_list: str
  ):
    self._dataframe = dataframe.fillna("-")
    self._rec_list = rec_list
    self._user_column = user_column
    
    self._df_metrics = None
    self._df_reclist = None
    self.fit(self._dataframe, self._user_column, self._rec_list)

  def fit(
      self,
      df: pd.DataFrame, 
      user_column: str,
      rec_list: str
  ) -> pd.DataFrame:

    self._df_reclist = df[[user_column, rec_list]]

  def cum_exposure(self, k: int = 10):
    df = self._df_reclist.copy()
    df[self._rec_list] = df[self._rec_list].apply(lambda l: l[:k])
    df['pos'] = [list(range(k)) for i in range(len(df))]


    total_exposure = len(self._dataframe) * k

    # Explode reclist
    df_sup_per_user = df.set_index([self._user_column])\
                          .apply(pd.Series.explode)\
                          .reset_index()
    df_sup_per_user['count'] = 1
    print(df_sup_per_user)

    # Mean Score pivot per column
    df_mean_scores_per_column = pd.pivot_table(df_sup_per_user, 
                                            index=self._rec_list, 
                                            columns='pos', 
                                            values='count', aggfunc=np.sum).fillna(0)
    print(df_mean_scores_per_column)
    print(df_mean_scores_per_column.values.max())


    df_sup_exp = df_sup_per_user.groupby(self._rec_list).count()\
                  .sort_values(self._user_column)

    #self._df_sup = np.unique(df_sup_per_user[rec_list])

    return df_sup_exp

  def metric(self):
    '''
    Rank Metrics - 
    '''
    return self.cum_exposure(10)

  def ndce_at_k(self, supp: str, k: int):
    '''
    Normalized Discounted Cumulative Exposure (NDCE)
    '''
    df = self._df_reclist.copy()
    df['item_column'] = supp

    with Pool(os.cpu_count()) as p:
      print("Creating the relevance lists...")
      df["relevance_list"] = list(
          tqdm(
              p.starmap(
                  _create_relevance_list,
                  zip(
                      df[self._rec_list],
                      df['item_column']
                  ),
              ),
              total=len(df),
          )
      )

      print("Calculating ndce@k...")
      df['ndce@{}'.format(k)] = list(
          tqdm(
              p.map(functools.partial(ndcg_at_k, k=k), df["relevance_list"]),
              total=len(df),
          )
      )

    return df['ndce@{}'.format(k)].mean()

  def show(self, **kwargs):
    '''
    '''    

    top_k  = 10 if 'top_k' not in kwargs else kwargs['top_k']
    title  = "Exposure" if 'title' not in kwargs else kwargs['title']

    df     = self.cum_exposure(top_k)
    data   = []
    
    exp_cum = list(df[self._user_column].cumsum())/df[self._user_column].sum()

    data.append(
        go.Scatter(
            name="Reclist",
            x=np.array(list(range(len(exp_cum))))*100/len(exp_cum),
            y=exp_cum*100,
        )
    )  

    data.append(
        go.Scatter(
            name="Equality",
            x=list(range(100)),
            y=list(range(100)),
            mode='markers',
            marker = dict(
                      color = 'rgb(128, 128, 128)',
                      size = 2,
                      opacity = 0.7
                    ),            
        )
    )      

    fig = go.Figure(data=data)

    # Change the bar mode
    fig.update_layout(
        template=TEMPLATE,
        legend_orientation="h",
        xaxis_title="% Producers",
        yaxis_title="% Exposure",
        legend=dict(y=-0.2),
        title=title,
    )

    # fig.add_annotation(
    #     x=0.8,
    #     y=0.1,
    #     xref="x",
    #     yref="y",
    #     text="Max K-S: {}".format(self.metric().round(3)),
    #     showarrow=False,
    #     font=dict(
    #         family="Courier New, monospace",
    #         size=12,
    #         color="#ffffff"
    #         ),
    #     align="center",
    #     bordercolor="#c7c7c7",
    #     borderwidth=1,
    #     borderpad=4,
    #     bgcolor="#ff7f0e",
    #     opacity=0.8
    # )

    return fig