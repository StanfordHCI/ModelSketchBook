# SKETCH

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np
from typing import List

from msb_enums import OutputType, idMode, ModelType, SketchSortMode
from SketchBook import SketchBook
from Result import Result
import helper_functions as h


# Class definition of a sketch
# - id: unique identifier string
# - concepts: list of IDs of concepts that the sketch will aggregate
# - input_fields: set of input fields that the specified concepts draw from
# - model_type: the model type that will aggregate the specified concepts
# - output_type: the type of output that the sketch will provide (ex: binary or continouous scores)
# - threshold: the threshold at which to binarize the sketch model predictions
# - cached_results: dict mapping from Dataset ID to Result
# - cached_perf: dict mapping from Dataset ID to performance dictionary (key=metric name, value = metric value)
class Sketch:
    def __init__(
        self,
        sb: SketchBook,
        concepts: List[str],
        model_type: ModelType,
        sort_mode: SketchSortMode = SketchSortMode.SketchPred,
        output_type: OutputType = OutputType.Continuous,
        threshold: float = 0.5,
    ):
        self.sb = sb
        self.id = h.create_id(sb, idMode.Sketch)
        self.concepts = concepts
        input_fields = [self.sb.concepts[c_id].input_field for c_id in self.concepts]
        self.input_fields = set([f for f in input_fields if f is not None])
        self.model_type = model_type
        self.sort_mode = sort_mode
        self.output_type = output_type
        self.threshold = threshold
        self.cached_results = {}
        self.cached_perf = {}
        self.model = None

        self.sb.add_sketch(self, self.id)  # add sketch to sketchbook

    def _get_perf_summary(self, dataset_id, preds, print_results=True):
        y_true, y_pred = self.sb.get_pred_and_ground_truth(preds, dataset_id)
        y_pred_bin = [0 if rating < self.threshold else 1 for rating in y_pred]
        y_true_bin = [0 if rating < self.threshold else 1 for rating in y_true]

        # Sketch perf
        perf = {}
        perf["mae"] = mean_absolute_error(y_true, y_pred)
        perf["acc"] = accuracy_score(y_true_bin, y_pred_bin)
        perf["f1"] = f1_score(y_true_bin, y_pred_bin)
        perf["prec"] = precision_score(y_true_bin, y_pred_bin)
        perf["recall"] = recall_score(y_true_bin, y_pred_bin)
        self.cached_perf[dataset_id] = perf

        def round_res(s):
            return np.round(s, 2)

        if print_results:
            print("\nPerformance summary:")
            print(f'\tMAE (Mean Absolute Error): {round_res(perf["mae"])}')
            print(f'\tAccuracy: {round_res(perf["acc"])}')
            print(f'\tF1 Score: {round_res(perf["f1"])}')
            print(f'\tPrecision: {round_res(perf["prec"])}')
            print(f'\tRecall: {round_res(perf["recall"])}')

    def _show_perf(self, dataset_id, preds):
        # Optionally show performance if there are ground truth labels
        dataset = self.sb.datasets[dataset_id]
        if dataset.labeled:
            self._get_perf_summary(dataset_id, preds)

    def _get_manual_linreg(self, x, weights):
        products = np.array(x) * np.array(weights)
        pred = np.sum(products)
        return pred

    # Internal helper function to calculate predictions for a given sketch
    def get_preds(self, dataset_id):
        model = self.model
        X, item_ids = self.sb.get_input_arrs(self.concepts, dataset_id)
        if self.model_type == ModelType.LogisticRegression:
            preds = {
                item_id: model.predict_proba([X[i]])[0][1]
                for i, item_id in enumerate(item_ids)
            }
        elif self.model_type in [
            ModelType.LinearRegression,
            ModelType.MLP,
            ModelType.DecisionTree,
            ModelType.RandomForest,
        ]:
            preds = {
                item_id: model.predict([X[i]])[0] for i, item_id in enumerate(item_ids)
            }
        elif self.model_type == ModelType.ManualLinear:
            # Special case: model stores manual weights; we manually calculate score based on these weights
            weights = model
            ordered_weights = [weights[c_id] for c_id in self.concepts]
            preds = {
                item_id: self._get_manual_linreg(X[i], ordered_weights)
                for i, item_id in enumerate(item_ids)
            }
        else:
            raise Exception(
                f"The `{self.model_type}` model type has not yet been implemented."
            )

        res = Result(self.id, preds)
        self.cached_results[dataset_id] = res

        self._show_perf(dataset_id, preds)
        return res

    def _binarize_ground_truth(self, y):
        return [1 if score >= self.threshold else 0 for score in y]

    def _readable_concept(self, concept_id):
        c = self.sb.concepts[concept_id]
        return f"{concept_id}: {c.concept_term} ({c.input_field})"

    def _readable_all_concepts(self):
        concept_strs = [self._readable_concept(c_id) for c_id in self.concepts]
        return ",".join(concept_strs)

    def _get_model_summary(self, dataset_id):
        assert (
            self.model is not None
        ), "The model summary is only available if a model has been trained. Please run the `train()` function on a training dataset first."

        if self.model_type == ModelType.LinearRegression:
            model = self.model
            weights = {
                self.sb._get_concept_term(concept_id): model.coef_[i]
                for i, concept_id in enumerate(self.concepts)
            }
            intercept = model.intercept_

            # Weights sorted by absolute value, high to low
            weights_ranked = sorted(
                weights.items(), key=lambda x: abs(x[1]), reverse=True
            )
            weights_ranked = [concept_term for concept_term, _ in weights_ranked]

            weights_sign = {
                "negative": [
                    concept_term
                    for concept_term, weight in weights.items()
                    if weight < 0
                ],
                "positive": [
                    concept_term
                    for concept_term, weight in weights.items()
                    if weight >= 0
                ],
            }

            gt_corr = self.sb._get_concept_gt_correlations(
                concept_ids=self.concepts, dataset_id=dataset_id
            )

            return weights, intercept, weights_ranked, weights_sign, gt_corr

    # Main function to train the sketch model on a specified dataset
    def train(self, dataset_id, show_params=True, manual_weights=None):
        X, _ = self.sb.get_input_arrs(self.concepts, dataset_id)
        y = self.sb.get_ground_truth_arr(dataset_id)

        if self.model_type == ModelType.LinearRegression:
            model = LinearRegression().fit(X, y)
            # TODO: helper function to view weights and intercept of model
            weights = {
                concept_id: model.coef_[i] for i, concept_id in enumerate(self.concepts)
            }
            intercept = model.intercept_
            if show_params:
                print("\nModel details:")
                print(f"\tIntercept: {np.round(intercept, 2)}")
                print("\tWeights:")
                for concept_id, weight in weights.items():
                    print(
                        f"\t\t{self._readable_concept(concept_id)}: {np.round(weight, 2)}"
                    )
        elif self.model_type == ModelType.MLP:
            y = self._binarize_ground_truth(y)
            model = MLPClassifier(
                random_state=0, max_iter=1200, hidden_layer_sizes=[10, 5]
            ).fit(X, y)
        elif self.model_type == ModelType.LogisticRegression:
            y = self._binarize_ground_truth(y)
            model = LogisticRegression(random_state=0, class_weight="balanced").fit(
                X, y
            )
        elif self.model_type == ModelType.RandomForest:
            y = self._binarize_ground_truth(y)
            model = RandomForestClassifier(n_estimators=10)
            model = model.fit(X, y)
        elif self.model_type == ModelType.DecisionTree:
            y = self._binarize_ground_truth(y)
            model = tree.DecisionTreeClassifier()
            model = model.fit(X, y)
        elif self.model_type == ModelType.ManualLinear:
            # Special case: model stores manual weights
            model = manual_weights
        else:
            raise Exception(
                f"The `{self.model_type}` model type has not yet been implemented."
            )

        self.model = model

    # Main function to evaluate a trained sketch model on a specified dataset
    def eval(self, dataset_id):
        if self.model is None:
            raise Exception(
                "The sketch model has not yet been trained on a dataset. Please run the `train()` function on a training dataset before evaluating on a dataset."
            )

        if dataset_id in self.cached_results:
            res = self.cached_results[dataset_id]
            self._show_perf(dataset_id, res.preds)
            return res

        return self.get_preds(dataset_id)

    def _get_vis_df(self, dataset_id):
        names_to_preds = {}
        # Fetch concept predictions
        concept_results = {
            c_id: self.sb.concepts[c_id].run(dataset_id) for c_id in self.concepts
        }
        names_to_preds = {c_id: res.preds for c_id, res in concept_results.items()}

        # Fetch sketch predictions
        sketch_res = self.eval(dataset_id)
        sketch_preds = sketch_res.preds
        sketch_pred_col = "sketch_pred"
        names_to_preds[sketch_pred_col] = sketch_preds

        # Join to form df
        df = self.sb.join_multi_preds(dataset_id, names_to_preds)
        ground_truth_col = self.sb.datasets[dataset_id].ground_truth

        # Filter columns to show: input, concept scores, sketch preds, gt ratings
        input_fields = list(self.input_fields)
        if ground_truth_col is not None:
            output_fields = list(names_to_preds.keys()) + [ground_truth_col]
        else:
            output_fields = list(names_to_preds.keys())
        cols_to_show = input_fields + output_fields + ["msb_item_id"]
        df = df[cols_to_show]

        return df

    def _style_cols(self, df, dataset_id, is_styled):
        # Add input columns
        if is_styled:
            df = self.sb.style_input_fields(df, dataset_id, self.input_fields)
            df = df.drop(columns=["msb_item_id"])

        # Add concept columns
        binary_fields, contin_fields = self.sb.sep_concept_col_types(self.concepts)

        # Add sketch pred column
        sketch_pred_col = "sketch_pred"
        output_type = self.output_type
        if output_type == OutputType.Binary:
            binary_fields.append(sketch_pred_col)
        else:
            contin_fields.append(sketch_pred_col)

        # Add GT column
        ground_truth_col = self.sb.datasets[dataset_id].ground_truth
        diff_fields = []
        if ground_truth_col is not None:
            # Add diff column
            diff_col = f"diff ({sketch_pred_col} - {ground_truth_col})"
            df[diff_col] = df[sketch_pred_col] - df[ground_truth_col]
            if output_type == OutputType.Binary:
                binary_fields.append(ground_truth_col)
                diff_fields.append(diff_col)
            else:
                contin_fields.append(ground_truth_col)
                diff_fields.append(diff_col)

        # Sort by specified column
        sort_key = None
        if self.sort_mode == SketchSortMode.SketchPred:
            sort_col = sketch_pred_col
        elif ground_truth_col is None:
            # Default to sketch-pred sorting if there is no ground truth
            sort_col = sketch_pred_col
            print(
                f"Dataset `{dataset_id}` does not have a ground-truth column. Sorting by the sketch prediction value instead."
            )
        elif self.sort_mode == SketchSortMode.GroundTruth:
            sort_col = ground_truth_col
        elif self.sort_mode == SketchSortMode.Diff:
            sort_col = diff_col
        elif self.sort_mode == SketchSortMode.AbsDiff:
            sort_col = diff_col
            sort_key = abs

        df = df.sort_values(by=sort_col, ascending=False, key=sort_key)

        if is_styled:
            df = self.sb.style_output_fields(
                df, binary_fields, contin_fields, diff_fields
            )
        return df

    def visualize(self, dataset_id, is_styled=True):
        df = self._get_vis_df(dataset_id)
        df = self._style_cols(df, dataset_id, is_styled)
        return df
