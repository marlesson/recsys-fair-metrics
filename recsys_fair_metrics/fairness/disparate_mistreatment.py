import pandas as pd
import numpy as np

np.seterr(divide="ignore", invalid="ignore")
from typing import List, Dict, Any
from sklearn.metrics import confusion_matrix
from recsys_fair_metrics.util.util import mean_confidence_interval
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


class DisparateMistreatment(object):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        supp_metadata: pd.DataFrame,
        metric: str,
        column: str,
        item_column: str,
        prediction_key: str,
    ):
        self._dataframe = dataframe.fillna("-")
        self._supp_metadata = supp_metadata
        self._column = [column]
        self._metric = metric
        self._item_column = item_column
        self._prediction_key = prediction_key

        self._df_metrics = None
        self.fit(
            self._dataframe, self._column, self._item_column, self._prediction_key
        )

    def fit(
        self,
        df: pd.DataFrame,
        sub_keys: List[str],
        item_column: str,
        prediction_key: str,
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        if self._supp_metadata is not None:
            meta = self._supp_metadata[[self._item_column, self._column[0]]].drop_duplicates().fillna("-")
            df   = df.merge(meta, on=self._item_column, how='left', suffixes=("", "_"))
        
        for sub_key in sub_keys:
            subs = df[sub_key].unique()

            for sub in subs:
                sub_df = df[df[sub_key] == sub]
                y_true, y_pred = (
                    sub_df[item_column].astype(str),
                    sub_df[prediction_key].astype(str),
                )

                cnf_matrix = confusion_matrix(y_true, y_pred)

                num_positives = np.sum(np.diag(cnf_matrix))
                num_negatives = np.sum(cnf_matrix) - num_positives

                fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
                fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
                tp = np.diag(cnf_matrix)
                tn = cnf_matrix.sum() - (fp + fn + tp)

                fp = fp.astype(float)
                fn = fn.astype(float)
                tp = tp.astype(float)
                tn = tn.astype(float)

                # Sensitivity, hit rate, recall, or true positive rate
                tpr = tp / (tp + fn)
                # Specificity or true negative rate
                tnr = tn / (tn + fp)
                # Precision or positive predictive value
                ppv = tp / (tp + fp)
                # Negative predictive value
                npv = tn / (tn + fn)
                # Fall out or false positive rate
                fpr = fp / (fp + tn)
                # False negative rate
                fnr = fn / (tp + fn)
                # False discovery rate
                fdr = fp / (tp + fp)
                # positive rate
                pr = (tp + fp) / (tp + fp + fn + tn)
                # positive rate
                nr = (tn + fn) / (tp + fp + fn + tn)

                # Overall accuracy
                acc = (tp + tn) / (tp + fp + fn + tn)

                # Balanced Accuracy (BA)
                bacc = (tpr + tnr) / 2

                # print(classification_report(y_true,y_pred))
                fpr, fpr_c = mean_confidence_interval(fpr)
                fnr, fnr_c = mean_confidence_interval(fnr)
                tpr, tpr_c = mean_confidence_interval(tpr)
                tnr, tnr_c = mean_confidence_interval(tnr)
                pr, pr_c = mean_confidence_interval(pr)
                nr, nr_c = mean_confidence_interval(nr)
                acc, acc_c = mean_confidence_interval(acc)
                bacc, bacc_c = mean_confidence_interval(bacc)

                rows.append(
                    {
                        "sub_key": sub_key,
                        "sub": sub,
                        "total_class": len(tp),
                        "false_positive_rate": fpr,
                        "false_positive_rate_C": fpr_c,
                        "false_negative_rate": fnr,
                        "false_negative_rate_C": fnr_c,
                        "true_positive_rate": tpr,
                        "true_positive_rate_C": tpr_c,
                        "true_negative_rate": tnr,
                        "true_negative_rate_C": tnr_c,
                        "positive_rate": pr,
                        "positive_rate_C": pr_c,
                        "negative_rate": nr,
                        "negative_rate_C": nr_c,
                        "accuracy": acc,
                        "accuracy_C": acc_c,
                        "balance_accuracy": bacc,
                        "balance_accuracy_C": bacc_c,
                        "total_positives": num_positives,
                        "total_negatives": num_negatives,
                        "total_individuals": num_positives + num_negatives,
                    }
                )
        
        self._df_metrics = pd.DataFrame(data=rows).sort_values(["sub_key", "sub"])
        self._df_metrics = self._df_metrics[
            [
                "sub_key",
                "sub",
                "true_positive_rate",
                "true_positive_rate_C",
                "total_individuals",
            ]
        ]
        self._df_metrics["feature"] = (
            self._df_metrics["sub_key"] + "@" + self._df_metrics["sub"].astype(str)
        )
        self._df_metrics = self._df_metrics.set_index("feature")
        # print(self._df_metrics)

    def metrics(self):
        return self._df_metrics

    def show(self, **kwargs):
        df = self._df_metrics
        metric = self._metric
        data = []

        title = (
            "Disparate Mistreatment: " + self._column[0]
            if "title" not in kwargs
            else kwargs["title"]
        )

        data.append(
            go.Bar(
                y=df.index,
                x=df[metric],
                orientation="h",
                error_x=dict(type="data", array=df[metric + "_C"])
                if metric + "_C" in df.columns
                else {},
                marker={"color": list(range(len(df.index))), "colorscale": "Tealgrn"},
            )
        )  # Plotly3

        fig = go.Figure(data=data)
        # Change the bar mode
        fig.update_layout(
            template=TEMPLATE,
            legend_orientation="h",
            xaxis_title=metric,
            xaxis_range=(0, np.max([1, df[metric].max()])),
            legend=dict(y=-0.2),
            title=title,
        )

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
                    x0=df[metric].mean(),
                    x1=df[metric].mean(),
                )
            ]
        )

        fig.add_annotation(
            x=np.max([1, df[metric].max()]),
            y=0,
            xref="x",
            yref="y",
            text="Max Diff: {}".format(self.metric()[self._metric].round(3)),
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

    def metric(self):
        """
        Max Diff
        """
        diffs = []
        for k in list(self._df_metrics[self._metric].index):
            for i in list(self._df_metrics[self._metric].index):
                e = np.abs(
                    self._df_metrics[self._metric][k]
                    - self._df_metrics[self._metric][i]
                )
                diffs.append(e)
        return {self._metric: np.max(diffs)}
