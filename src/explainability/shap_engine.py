import shap
import numpy as np

class SHAPEngine:
    def __init__(self, model, data, feature_names):
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer(data)

    def get_patient_explanation(self, idx):
        sv = self.shap_values[idx]

        # ALWAYS handle binary classification safely
        if isinstance(sv, shap.Explanation):
            sv = sv.values

        if sv.ndim == 2:
            sv = sv[:, 1]   # class 1 (disease)

        explanation = shap.Explanation(
            values=sv,
            base_values=self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value,
            data=self.data.iloc[idx].values,
            feature_names=self.feature_names
        )

        return explanation, sv