# SKETCHBOOK

from PIL import Image
import requests
from io import BytesIO
import base64
import numpy as np
from typing import List, Dict

from .msb_enums import InputType, idMode
from .Dataset import Dataset
from .helper_functions import *


# Class definition of a sketchbook
# - goal: description of modeling task, objective
# - datasets (user input): list of already-created Datasets
# - datasets (sketchbook attribute): dict mapping id to a Dataset
# - schema: dict of inputs to be used in concepts in sketches, in the form { name of input column : input type }
# - default_dataset_id: string indicating the ID of the dataset that should be used by default if none is specified
# - concepts: dict mapping id to a Concept
# - sketches: dict mapping id to a Sketch
# - notes: dict mapping id to note for a Concept or Sketch
class SketchBook:
    def __init__(
        self,
        goal: str,
        schema: dict,
    ):
        self.goal = goal
        self.datasets = {}
        self.default_dataset_id = None
        ground_truth = self._validate_schema(schema)
        self.schema = schema
        self.ground_truth = ground_truth

        self.concepts = {}
        self.sketches = {}
        self.notes = {}

    def _validate_schema(self, schema):
        # There can only be one ground-truth column
        ground_truth_cols = []
        for input_field, input_type in schema.items():
            if input_type == InputType.GroundTruth:
                ground_truth_cols.append(input_field)

        assert (
            len(ground_truth_cols) == 1
        ), f"Only one ground truth column is allowed; you have specified the following ground truth columns: {ground_truth_cols}"
        return ground_truth_cols[0]

    def _validate_dataframe(self, df):
        # Check that the provided dataframe matches the sketchbook schema
        for input_field, input_type in self.schema.items():
            if input_type != InputType.GroundTruth:
                # Columns other than the ground truth column are required
                assert (
                    input_field in df.columns
                ), f"The input field `{input_field}` is not in the provided dataset. Please update your dataset or schema accordingly."

    def add_dataset(
        self,
        df,
        default=False,
        cache_data=True,
    ):
        id_string = create_id(self, idMode.Dataset)
        self._validate_dataframe(df)

        if self.ground_truth in df.columns:
            ground_truth = self.ground_truth
        else:
            ground_truth = None

        dataset = Dataset(id_string, df, ground_truth)
        self.datasets[dataset.id] = dataset
        if default:
            self.default_dataset_id = dataset.id

        if cache_data:
            print("Caching data...")
            for input_field, input_type in self.schema.items():
                if (input_type == InputType.Image) or (
                    input_type == InputType.ImageLocal
                ):
                    # Load and cache images
                    self._cache_image_field(dataset, input_field, input_type)
            print("Done caching!")

    def _cache_image_field(
        self,
        dataset: Dataset,
        input_field: str,
        input_type: InputType,
        max_width: int = 250,
    ):
        item_inputs = dataset.df[input_field].tolist()
        item_ids = dataset.get_item_ids()
        images = {}
        images_html = {}

        for idx, item_input in enumerate(item_inputs):
            item_id = item_ids[idx]
            try:
                if input_type == InputType.ImageLocal:
                    # 'url' is a local file path
                    img = Image.open(item_input)
                    img_html = (
                        f'<img src="{item_input}" style="max-width: {max_width}px;">'
                    )
                else:
                    # Fetch the image from the provided remote URL
                    response = requests.get(item_input)
                    img = Image.open(BytesIO(response.content))
                    img = rescale_img(img)

                    # Save image string
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("ascii")
                    img_html = f'<img src="data:image/png;base64,{img_str}" style="max-width: {max_width}px;">'

                images[item_id] = img
                images_html[item_id] = img_html
            except Exception as e:
                print("Error in loading", item_input, e)
                return
        dataset.cached_images[input_field] = images
        dataset.cached_images_html[input_field] = images_html

    def join_preds(
        self,
        dataset_id: str,
        preds: Dict[int, float],
        pred_col_name: str = "pred",
    ):
        dataset = self.datasets[dataset_id]
        df = dataset.df.copy()
        # Join preds with df on msb_item_id
        item_ids = dataset.get_item_ids()
        df[pred_col_name] = [preds[item_id] for item_id in item_ids]
        return df

    def join_multi_preds(
        self,
        dataset_id: str,
        names_to_preds: Dict[str, Dict[int, float]],
    ):
        dataset = self.datasets[dataset_id]
        df = dataset.df.copy()
        # Join preds with df on msb_item_id
        item_ids = dataset.get_item_ids()
        for col_name, cur_preds in names_to_preds.items():
            df[col_name] = [cur_preds[item_id] for item_id in item_ids]
        return df

    # Prepare the input array X for a sketch model
    # For each row in the dataset, get list of all concept scores in order
    def get_input_arrs(
        self,
        concept_ids: List[str],
        dataset_id: str,
    ):
        dataset = self.datasets[dataset_id]
        concept_results = [self.concepts[c_id].run(dataset_id) for c_id in concept_ids]
        concept_preds = [res.preds for res in concept_results]

        # Gather all input arrays
        item_ids = dataset.get_item_ids()
        input_arrs = [
            [concept_pred[item_id] for concept_pred in concept_preds]
            for item_id in item_ids
        ]
        return input_arrs, item_ids

    # Prepare the output array y for a sketch model
    # Returns a list of the specified ground truth valuse for the dataset
    def get_ground_truth_arr(
        self,
        dataset_id: str,
    ):
        dataset = self.datasets[dataset_id]
        df = dataset.df
        if dataset.ground_truth is None:
            raise Exception(
                f"Dataset {dataset_id} has no specified ground truth column."
            )
        return np.asarray(df[dataset.ground_truth].tolist())

    def get_pred_and_ground_truth(self, preds, dataset_id: str):
        dataset = self.datasets[dataset_id]

        # Gather all input arrays
        item_ids = dataset.get_item_ids()
        y_pred = [preds[item_id] for item_id in item_ids]
        y_true = self.get_ground_truth_arr(dataset_id)

        return y_true, y_pred

    def _get_concept_term(self, concept_id):
        return self.concepts[concept_id].concept_term

    def _get_concept_preds_df(self, concept_ids, dataset_id):
        # Get concept predictions
        concept_results = {
            c_id: self.concepts[c_id].run(dataset_id) for c_id in concept_ids
        }
        names_to_preds = {c_id: res.preds for c_id, res in concept_results.items()}
        df = self.join_multi_preds(dataset_id, names_to_preds)
        return df

    # Calculates Pearson correlation coefficients for all concepts versus the ground truth label
    # Returns a dictionary with key=concept_term and value=correlation
    def _get_concept_gt_correlations(
        self, concept_ids, dataset_id, concept_term_key=False, sort=False
    ):
        concept_arrs, item_ids = self.get_input_arrs(concept_ids, dataset_id)
        concept_arrs = np.transpose(np.array(concept_arrs))
        gt_arr = self.get_ground_truth_arr(dataset_id)

        corr = {}
        for concept_arr, concept_id in zip(concept_arrs, self.concepts):
            r = np.corrcoef(concept_arr, gt_arr)[0, 1]
            if concept_term_key:
                concept_term = self._get_concept_term(concept_id)
                corr[concept_term] = r
            else:
                corr[concept_id] = r

        # Sort by correlation absolute value, high to low
        if sort:
            corr = sorted(corr.items(), key=lambda x: abs(x[1]), reverse=True)
        return corr

    # Given a dataframe with concept scores and a ground truth column, For each provided concept, calculate the difference between that concept's scores and the ground truth of the provided dataset
    def _get_concept_gt_diffs(self, df, concept_ids, dataset_id):
        ground_truth_col = self.datasets[dataset_id].ground_truth
        diff_cols = [f"diff ({c_id} - {ground_truth_col})" for c_id in concept_ids]
        for c_id, diff_col in zip(concept_ids, diff_cols):
            concept_scores = df[c_id]
            gt_scores = df[ground_truth_col]
            # TODO: handle binary case
            df[diff_col] = concept_scores - gt_scores
        return df, diff_cols

    def style_input_field(self, df, dataset_id, input_field):
        input_type = self.schema[input_field]
        if input_type == InputType.Image or input_type == InputType.ImageLocal:
            # Render images
            item_ids = df["msb_item_id"].tolist()
            images_html = self.datasets[dataset_id].cached_images_html[input_field]
            df[input_field] = [images_html[item_id] for item_id in item_ids]
        return df

    def style_input_fields(self, df, dataset_id, input_fields):
        for input_field in input_fields:
            df = self.style_input_field(df, dataset_id, input_field)
        return df

    def sep_concept_col_types(self, concept_ids):
        binary_fields = []
        contin_fields = []
        for c_id in concept_ids:
            is_binary = self.concepts[c_id]._has_binary_output()
            if is_binary:
                binary_fields.append(c_id)
            else:
                contin_fields.append(c_id)
        return binary_fields, contin_fields

    def style_output_fields(self, df, binary_fields, contin_fields, diff_fields):
        # Separated by binary and continouous fields since styling has to be chained into a single df.style call
        if len(binary_fields) > 0 and len(contin_fields) > 0:
            df = (
                df.style.apply(color_t_f, subset=binary_fields)
                .apply(color_magnitude, subset=contin_fields)
                .apply(color_pos_neg, subset=diff_fields)
            )
        elif len(binary_fields) > 0:
            df = df.style.apply(color_t_f, subset=binary_fields).apply(
                color_pos_neg, subset=diff_fields
            )
        elif len(contin_fields) > 0:
            df = df.style.apply(color_magnitude, subset=contin_fields).apply(
                color_pos_neg, subset=diff_fields
            )
        else:
            return df

        df = df.set_table_styles(
            [
                dict(
                    selector="th",
                    props=[("min-width", "100px"), ("word-break", "break-word")],
                )
            ]
        ).set_sticky(
            axis=1
        )  # sticky header
        return df

    def add_concept(
        self,
        concept,
        id,
    ):
        self.concepts[id] = concept

    def get_concept(
        self,
        id,
    ):
        return self.concepts[id]

    def add_sketch(
        self,
        sketch,
        id,
    ):
        self.sketches[id] = sketch

    def get_sketch(
        self,
        id,
    ):
        return self.sketches[id]
