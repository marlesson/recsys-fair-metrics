import pandas as pd
from recsys_fair_metrics.util.util import parallel_literal_eval
from recsys_fair_metrics.fairness.disparate_mistreatment import DisparateMistreatment
from recsys_fair_metrics.fairness.disparate_treatment import DisparateTreatment
from recsys_fair_metrics.relevance.rank_metrics import RelevanceRank
from recsys_fair_metrics.exposure.exposure_metrics import ExposureMetric
from typing import List, Dict, Any


class RecsysFair(object):
    def __init__(
        self,
        df: pd.DataFrame,
        user_metadata: pd.DataFrame = None,
        supp_metadata: pd.DataFrame = None,
        user_column: str = "user_id",
        item_column: str = "item_id",
        reclist_column: str = "reclist_column",
        reclist_score_column: str = "reclist_score_column",
    ):
        self._user_column = user_column
        self._item_column = item_column
        self._reclist_column = reclist_column
        self._reclist_score_column = reclist_score_column
        self._supp_metadata = supp_metadata
        self._user_metadata = user_metadata
        self._dataframe = df.copy()
        self.fit()

    def fit(self):
        self._dataframe[self._reclist_column] = parallel_literal_eval(
            self._dataframe[self._reclist_column]
        )
        self._dataframe[self._reclist_score_column] = parallel_literal_eval(
            self._dataframe[self._reclist_score_column]
        )
        self._dataframe["first_rec"] = self._dataframe[self._reclist_column].apply(
            lambda l: l[0]
        )
        self._dataframe["first_recscore"] = self._dataframe[
            self._reclist_score_column
        ].apply(lambda l: l[0])

    def disparate_mistreatment(self, column: str):
        return DisparateMistreatment(
            dataframe=self._dataframe,
            metric="true_positive_rate",
            column=column,
            ground_truth_key=self._item_column,
            prediction_key="first_rec",
        )

    def disparate_treatment(self, column: str):
        return DisparateTreatment(
            dataframe=self._dataframe,
            metric="true_positive_rate",
            column=column,
            prediction_score_key=self._reclist_score_column,
            prediction_key=self._reclist_column,
        )

    def exposure(self, column: str, k: int = 10):
        return ExposureMetric(
            dataframe=self._dataframe,
            supp_metadata=self._supp_metadata,
            column=column,
            user_column=self._user_column,
            item_column=self._item_column,
            rec_list=self._reclist_column,
            k=k,
        )

    def relevance(self, metrics: List[str] = []):
        return RelevanceRank(
            dataframe=self._dataframe,
            metrics=metrics,
            item_column=self._item_column,
            rec_list=self._reclist_column,
        )
