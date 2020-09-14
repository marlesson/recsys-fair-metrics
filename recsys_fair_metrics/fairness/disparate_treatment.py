import pandas as pd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix
from scipy.stats import norm
from recsys_fair_metrics.util.util import mean_confidence_interval
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp

TEMPLATE = "plotly_white"


class DisparateTreatment(object):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        metric: str,
        column: str,
        prediction_score_key: str,
        prediction_key: str,
    ):
        self._dataframe = dataframe.fillna("-")
        self._column = column
        self._metric = metric
        self._prediction_score_key = prediction_score_key
        self._prediction_key = prediction_key

        self._df_scores = None
        self._df_mean_scores = None
        self._df_mean_scores_per_column = None

        self.fit(
            self._dataframe,
            self._column,
            self._prediction_score_key,
            self._prediction_key,
        )

    def fit(
        self,
        df: pd.DataFrame,
        column: List[str],
        prediction_score_key: str,
        prediction_key: str,
    ) -> pd.DataFrame:

        df = df[[column, prediction_key, prediction_score_key]]
        df = df.set_index([column]).apply(pd.Series.explode).reset_index().fillna("-")
        # from IPython import embed; embed()
        def confidence(x):
            return mean_confidence_interval(x)[1]

        df_mean = (
            df.groupby([prediction_key, column]).agg(
                count=(prediction_score_key, "count"),
                mean_rhat_score=(prediction_score_key, "mean"),
                confidence=(prediction_score_key, confidence),
            )
        ).reset_index()  # .rename(columns={item_column: 'count', first_recscore_column: 'mean_rhat_score'})

        # Mean Score List
        df_mean = self.filter_treatment_df(
            df_mean, prediction_key, column, min_size=10
        ).sort_values("mean_rhat_score")

        # Mean Score pivot per column
        df_mean_scores_per_column = df_mean.pivot(
            index=prediction_key, columns=self._column, values=["mean_rhat_score"]
        )  # .fillna()

        self._sample_score = 10000
        self._df_scores = df
        self._df_mean_scores = df_mean
        self._df_mean_scores_per_column = df_mean_scores_per_column["mean_rhat_score"]

    def filter_treatment_df(self, df, rec_column, fairness_column, min_size=10):
        # Filter significance
        df = df[df["count"] >= min_size]

        # more them one group
        df_count = df.groupby([rec_column]).count()["count"].reset_index()
        df_count = df_count[
            df_count["count"] > 1
        ]  # == len(np.unique(df[fairness_column]))
        df = df[df[rec_column].isin(df_count[rec_column])]

        return df

    def metrics(self):
        return self._df_metrics

    def show(self, kind: str = "bar", **kwargs):
        if kind == "bar":
            return self.show_bar(**kwargs)
        elif kind == "ks":
            return self.show_ks(**kwargs)

    def show_bar(self, **kwargs):

        title = (
            "Disparate Treatment: " + self._column
            if "title" not in kwargs
            else kwargs["title"]
        )
        top_k = 10 if "top_k" not in kwargs else kwargs["top_k"]

        df = self._df_mean_scores
        column = self._column
        reclist_column = self._prediction_key
        data = []

        # Order by variance
        df_var = (
            df.groupby(reclist_column)
            .agg(var_mean_rhat_score=("mean_rhat_score", "var"))
            .sort_values("var_mean_rhat_score", ascending=False)
            .reset_index()
            .iloc[:top_k]
        )

        df = df.merge(df_var).sort_values("var_mean_rhat_score", ascending=False)

        y_sorted = list(
            df[[reclist_column, "var_mean_rhat_score"]]
            .drop_duplicates()
            .sort_values("var_mean_rhat_score", ascending=False)[reclist_column]
        )
        y_sorted.reverse()

        for group, rows in df.groupby(column):
            # rows = rows.sort_values('var_mean_rhat_score', ascending=False)
            data.append(
                go.Bar(
                    name=column + "." + str(group),
                    y=[str(a) for a in rows[reclist_column]],
                    x=rows["mean_rhat_score"],
                    orientation="h",
                    error_x=dict(type="data", array=rows["confidence"]),
                )
            )  # px.colors.sequential.Purp [i for i in range(len(rows))]

        fig = go.Figure(data=data)

        # Change the bar mode
        fig.update_layout(
            template=TEMPLATE,
            legend_orientation="h",
            xaxis_title="rhat_scores",
            legend=dict(y=-0.2),
            title=title,
        )
        fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": y_sorted})

        fig.update_layout(
            shapes=[
                dict(
                    type="line",
                    line=dict(
                        width=1,
                        dash="dot",
                    ),
                    yref="paper",
                    y0=0,
                    y1=1,
                    xref="x",
                    x0=df["mean_rhat_score"].mean(),
                    x1=df["mean_rhat_score"].mean(),
                )
            ]
        )

        return fig

    def show_ks(self, **kwargs):
        """
        In an ECDF, x-axis correspond to the range of values for variables
        and on the y-axis we plot the proportion of data points that are less
        than are equal to corresponding x-axis value.

        https://cmdlinetips.com/2019/05/empirical-cumulative-distribution-function-ecdf-in-python/
        """

        title = (
            "Disparate Treatment: " + self._column
            if "title" not in kwargs
            else kwargs["title"]
        )

        df = self._df_scores
        data = []
        column = self._column
        for group in np.unique(df[column].values):
            values = (
                df[df[column] == group]["action_scores"]
                .sample(self._sample_score, random_state=42, replace=True)
                .values
            )
            x, y = self.ecdf(values)
            data.append(
                go.Scatter(
                    name=column + "." + str(group),
                    y=y,
                    x=x,
                )
            )
        # fig = px.histogram(df, x='')
        fig = go.Figure(data=data)

        # Change the bar mode
        fig.update_layout(
            template=TEMPLATE,
            legend_orientation="h",
            xaxis_title="rhat_score",
            yaxis_title="ECDF",
            legend=dict(y=-0.2),
            title=title,
        )

        fig.add_annotation(
            x=0.8,
            y=0.1,
            xref="x",
            yref="y",
            text="Max K-S: {}".format(self.metric()["max_ks"].round(3)),
            showarrow=False,
            font=dict(family="Courier New, monospace", size=12, color="#ffffff"),
            align="center",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8,
        )

        return fig

    def ecdf(self, data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return (x, y)

    def metric(self):
        """
        Max KS Distance
        """
        df = self._df_scores
        columns = np.unique(df[self._column].values)
        ks_metrics = []
        for a in columns:
            for b in columns:
                sample_a = (
                    df[df[self._column] == a]["action_scores"]
                    .sample(self._sample_score, random_state=42, replace=True)
                    .values
                )
                sample_b = (
                    df[df[self._column] == b]["action_scores"]
                    .sample(self._sample_score, random_state=42, replace=True)
                    .values
                )
                ks_metrics.append(ks_2samp(sample_a, sample_b).statistic)

        # for a in self._df_mean_scores_per_column.columns:
        #     for b in self._df_mean_scores_per_column.columns:
        #         ks_metrics.append(ks_2samp(self._df_mean_scores_per_column[a], self._df_mean_scores_per_column[b]).statistic)

        # s = len(self._df_mean_scores_per_column.columns)
        return {"max_ks": np.max(ks_metrics)}
