import pandas as pd
import numpy as np


class FairnessEngine:
    def __init__(self, model, X, y, features, raw_df=None):
        """
        model: trained ML model
        X: feature matrix used for prediction
        y: target
        features: model feature names
        raw_df: full dataframe (includes age_group, sex, etc.)
        """
        self.model = model
        self.X = X.copy()
        self.y = y.reset_index(drop=True)
        self.features = features

        # IMPORTANT FIX: fairness uses raw dataset if available
        self.raw_df = raw_df if raw_df is not None else X.copy()

    # =========================
    # GROUP FAIRNESS ANALYSIS
    # =========================
    def evaluate_group(self, group_col):

        results = []

        # FIX: check in raw_df, not X
        if group_col not in self.raw_df.columns:
            raise KeyError(
                f"'{group_col}' not found. Available columns: {list(self.raw_df.columns)}"
            )

        for g in self.raw_df[group_col].dropna().unique():

            mask = self.raw_df[group_col] == g

            # prediction uses X only (safe ML design)
            X_g = self.X.loc[mask, self.features]
            y_g = self.y.loc[mask]

            if len(y_g) < 10:
                continue

            preds = self.model.predict(X_g)

            acc = (preds == y_g).mean()

            rec = (
                ((preds == 1) & (y_g == 1)).sum()
                / max((y_g == 1).sum(), 1)
            )

            results.append({
                "group": str(g),
                "size": len(y_g),
                "accuracy": float(acc),
                "recall": float(rec)
            })

        df = pd.DataFrame(results)

        if df.empty:
            return df, 0.0, 0.0

        recall_gap = df["recall"].max() - df["recall"].min()
        acc_gap = df["accuracy"].max() - df["accuracy"].min()

        return df, float(recall_gap), float(acc_gap)

    # =========================
    # COMPATIBILITY WRAPPER
    # =========================
    def evaluate(self, feature):
        return self.evaluate_group(feature)