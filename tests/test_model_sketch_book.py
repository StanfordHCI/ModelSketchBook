"""
ModelSketchBook tests
From parent directory: 
pytest tests/

To avoid showing warnings:
pytest tests/ --disable-warnings 

Test coverage:
pytest tests/ --cov=model_sketch_book

"""

import pandas as pd
import numpy as np
import unittest

import model_sketch_book as msb
from model_sketch_book.Concept import (
    Concept,
    ImageConcept,
    GPTTextConcept,
    LogicalConcept,
    KeywordConcept,
)
from model_sketch_book.Concept import (
    get_prompt,
    parse_gpt_response,
)
from model_sketch_book.Result import Result
from model_sketch_book.Sketch import Sketch

test_img_url = "https://www.w3schools.com/html/w3schools.jpg"
N_ROWS = 5


# End-to-end testing
def setup_sb():
    sb = msb.create_model_sketchbook(
        goal="Test MSB",
        schema={
            "img_url": msb.InputType.Image,
            "text": msb.InputType.Text,
            "overall_rating_score": msb.InputType.GroundTruth,
        },
        credentials={"organization": "org-HERE", "api_key": "sk-HERE"},
    )
    train_row_scores = [(0.01 * i) for i in range(N_ROWS)]
    train_rows = [
        [test_img_url, f"text_{i}", s] for i, s in enumerate(train_row_scores)
    ]
    train = pd.DataFrame(
        train_rows, columns=["img_url", "text", "overall_rating_score"]
    )
    sb.add_dataset(df=train, default=True, cache_data=False)

    test_row_scores = [(0.05 * i) for i in range(N_ROWS)]
    test_rows = [[test_img_url, f"text_{i}", s] for i, s in enumerate(test_row_scores)]
    test = pd.DataFrame(test_rows, columns=["img_url", "text", "overall_rating_score"])
    sb.add_dataset(df=test, cache_data=False)
    return sb


## CONCEPTS
class TestConcepts(unittest.TestCase):
    def test_concept_Image(self):
        sb = setup_sb()
        concept = ImageConcept(sb, "person", "img_url", msb.OutputType.Continuous)
        res = concept.run(dataset_id="d_0")

        assert (
            len(res.preds) == N_ROWS
        ), f"Expected {N_ROWS} rows, but observed {len(res.preds)} rows"
        for val in res.preds.values():
            assert (
                type(val) == float
            ), f"ImageConcept continuous predictions are type {type(val)} instead of float"

    def test_concept_Text(self):
        sb = setup_sb()
        concept = GPTTextConcept(sb, "person", "text", msb.OutputType.Continuous)
        res = concept.run(dataset_id="d_0")

        assert (
            len(res.preds) == N_ROWS
        ), f"Expected {N_ROWS} rows, but observed {len(res.preds)} rows"
        for val in res.preds.values():
            assert (type(val) == float) or (
                type(val) == np.float64
            ), f"GPTTextConcept continuous predictions are type {type(val)} instead of float"

    def test_concept_Logical(self):
        sb = setup_sb()

        preds = {}
        concept_ids = []
        for c_term in ["test", "testing"]:
            c, _ = msb.create_concept_model_internal(
                sb,
                concept_term=c_term,
                input_field="text",
                output_type=msb.OutputType.Binary,
                is_testing=True,  # Generates random preds
                is_styled=False,
            )
            preds[c.id] = c.fetch_from_cache(dataset_id="d_0").preds
            concept_ids.append(c.id)

        dataset = sb.datasets["d_0"]
        item_ids = dataset.get_item_ids()
        expected_preds_AND = {
            item_id: np.logical_and(
                preds[concept_ids[0]][item_id], preds[concept_ids[1]][item_id]
            )
            for item_id in item_ids
        }
        expected_preds_OR = {
            item_id: np.logical_or(
                preds[concept_ids[0]][item_id], preds[concept_ids[1]][item_id]
            )
            for item_id in item_ids
        }

        # Test AND operator
        concept = LogicalConcept(
            sb,
            concept_term=None,
            input_field=None,
            output_type=msb.OutputType.Binary,
            subconcept_ids=["c_0", "c_1"],
            operator="AND",
        )
        res = concept.run(dataset_id="d_0")
        observed_preds = res.preds

        assert (
            observed_preds == expected_preds_AND
        ), "Logical AND predictions did not match expected values"

        # Test OR operator
        concept = LogicalConcept(
            sb,
            concept_term=None,
            input_field=None,
            output_type=msb.OutputType.Binary,
            subconcept_ids=["c_0", "c_1"],
            operator="OR",
        )
        res = concept.run(dataset_id="d_0")
        observed_preds = res.preds

        assert (
            observed_preds == expected_preds_OR
        ), "Logical OR predictions did not match expected values"

    def test_concept_Keyword(self):
        sb = setup_sb()
        keywords = ["1", "3"]
        concept = KeywordConcept(
            sb=sb,
            concept_term=None,
            input_field="text",
            output_type=msb.OutputType.Binary,
            keywords=keywords,
            case_sensitive=False,
        )
        res = concept.run(dataset_id="d_0")
        observed_preds = res.preds

        dataset = sb.datasets["d_0"]
        item_ids = dataset.get_item_ids()
        texts = dataset.df["text"].tolist()
        expected_preds = {
            item_id: ("1" in text) or ("3" in text)
            for item_id, text in zip(item_ids, texts)
        }

        assert (
            observed_preds == expected_preds
        ), "Keyword predictions did not match expected values"

    ## GPT-related functions
    def test_get_prompt_mult(self):
        comments = [
            "testing 1",
            "testing 2",
            "testing 3",
        ]
        concept_term = "hateful"
        item_type = "comment"
        max_example_length = 200

        observed = get_prompt(
            comments, concept_term, max_example_length, item_type, truncate_item=True
        )

        comments_numbered = [f"{i + 1}: '{c}'" for i, c in enumerate(comments)]
        intro = f"Decide whether these comments are 'hateful' or 'not hateful'.\n\n"
        examples = "\n".join(comments_numbered)
        outro = "\n\ncomment results:"
        expected = intro + examples + outro
        assert observed == expected, f"Prompt did not match expected format."

    def test_get_prompt_single(self):
        comments = ["testing1"]
        concept_term = "hateful"
        item_type = "comment"
        max_example_length = 200

        observed = get_prompt(
            comments, concept_term, max_example_length, item_type, truncate_item=True
        )

        intro = f"Decide whether this comment is 'hateful' or 'not hateful'.\n\n"
        example = f"comment: {comments[0]}"
        outro = "\n\ncomment result:"
        expected = intro + example + outro
        assert observed == expected, "Prompt did not match expected format."

    def test_parse_gpt_response(self):
        concept_term = "hateful"
        separators = [":", "-", "."]
        for sep in separators:
            results_items = [
                f"1{sep} hateful",
                f"2{sep} not hateful",
                f"3{sep} hateful",
            ]
            results = "\n".join(results_items)
            observed_pred, observed_results_isolated = parse_gpt_response(
                results, concept_term
            )
            expected_pred = [True, False, True]
            expected_results_isolated = ["hateful", "not hateful", "hateful"]
            assert (
                observed_pred == expected_pred
            ), f"Parsed GPT response predictions did not match expected format (using separator `{sep}`)."
            assert (
                observed_results_isolated == expected_results_isolated
            ), f"Parsed GPT response result strings did not match expected format (using separator `{sep}`)."

    ## CONCEPT TUNING
    def test_tune_threshold(self):
        sb = setup_sb()
        concept = ImageConcept(sb, "person", "img_url", msb.OutputType.Continuous)
        item_ids = sb.datasets["d_0"].get_item_ids()
        preds = {item_ids[i]: (0.2 * i) for i in range(len(item_ids))}
        res = Result(concept.id, preds)

        threshold = 0.5
        observed_res = concept._tune_threshold(res, threshold)

        # concept._set_threshold(threshold)
        # observed_res = concept.tune(res)
        observed = observed_res.preds
        expected = {item_id: (p >= threshold) for item_id, p in res.preds.items()}
        assert (
            observed == expected
        ), "The thresholded predictions don't match the expected values."

    def test_tune_calibrate(self):
        sb = setup_sb()
        concept = ImageConcept(sb, "person", "img_url", msb.OutputType.Continuous)
        item_ids = sb.datasets["d_0"].get_item_ids()
        preds = {item_ids[i]: (0.2 * i) for i in range(len(item_ids))}
        res = Result(concept.id, preds)

        calib = [0.2, 0.8]
        observed_res = concept._tune_calibrate(res, calib)

        # concept._set_calib(calib)
        # observed_res = concept.tune(res)
        observed_preds = observed_res.preds
        observed = list(observed_preds.values())
        expected = [0.0, 0.0, (0.2 / 0.6), (0.4 / 0.6), 1.0]
        assert np.allclose(
            observed, expected
        ), "The calibrated predictions don't match the expected values."

    def test_tune_normalize(self):
        sb = setup_sb()
        concept = ImageConcept(sb, "person", "img_url", msb.OutputType.Continuous)
        item_ids = sb.datasets["d_0"].get_item_ids()
        preds = {item_ids[i]: (0.1 * (i + 1)) for i in range(len(item_ids))}
        res = Result(concept.id, preds)

        observed_res = concept._tune_normalize(res)

        # concept._set_normalize(True)
        # observed_res = concept.tune(res)
        observed_preds = observed_res.preds
        observed = list(observed_preds.values())
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert np.allclose(
            observed, expected
        ), "The normalized predictions don't match the expected values."


## SKETCHES
class TestSketches(unittest.TestCase):
    def get_concepts_and_preds(self, sb):
        # Create concepts (image, text)
        concepts = {}
        preds = {}

        item_ids = sb.datasets["d_0"].get_item_ids()
        concept = ImageConcept(sb, "a", "img_url", msb.OutputType.Continuous)
        concepts[concept.id] = concept
        vals = [0.6, -0.3, 0.54, 0.73, -0.62]
        preds[concept.id] = {item_ids[i]: vals[i] for i in range(N_ROWS)}
        res = Result(concept.id, preds[concept.id])
        concept.add_to_cache(dataset_id="d_0", res=res)

        concept = GPTTextConcept(sb, "b", "text", msb.OutputType.Continuous)
        concepts[concept.id] = concept
        vals = [-0.53, 0.64, 0.17, 0.38, -0.73]
        preds[concept.id] = {item_ids[i]: vals[i] for i in range(N_ROWS)}
        res = Result(concept.id, preds[concept.id])
        concept.add_to_cache(dataset_id="d_0", res=res)

        concept = GPTTextConcept(sb, "c", "text", msb.OutputType.Binary)
        concepts[concept.id] = concept
        vals = [0.25, 0.91, -0.58, 0.08, -0.39]
        preds[concept.id] = {item_ids[i]: vals[i] for i in range(N_ROWS)}
        res = Result(concept.id, preds[concept.id])
        concept.add_to_cache(dataset_id="d_0", res=res)

        return concepts, preds, item_ids

    def test_sketch_LinReg(self):
        # Test the LinearRegression sketch model type
        sb = setup_sb()
        concepts, preds, item_ids = self.get_concepts_and_preds(sb)

        # Create sketch
        sketch = Sketch(
            sb,
            concepts=[c.id for c in concepts.values()],
            model_type=msb.ModelType.LinearRegression,
            output_type=msb.OutputType.Continuous,
        )

        # Set ground-truth
        dataset = sb.datasets["d_0"]
        df = dataset.df
        weights = [0.1, 0.9, -0.3]
        intercept = 0.3
        x_arrs = [
            [preds[c_id][item_ids[i]] for c_id in sb.concepts.keys()]
            for i in range(len(item_ids))
        ]
        df[dataset.ground_truth] = [
            np.sum(np.array(x) * np.array(weights)) + intercept for x in x_arrs
        ]
        dataset.df = df
        sketch.train(dataset_id="d_0")
        df_out = sketch.visualize(dataset_id="d_0", is_styled=False)

        # Verify learned weights and intercept
        model = sketch.model
        obs_weights = [model.coef_[i] for i, c_id in enumerate(sb.concepts.keys())]
        obs_intercept = model.intercept_
        assert np.allclose(
            obs_weights, weights
        ), f"Learned LinearRegression weights differ from true weights."
        assert np.isclose(
            obs_intercept, intercept
        ), "Learned LinearRegression intercept differs from true intercept."

    def test_sketch_ManualLinear(self):
        # Test the ManualLinear sketch model type
        sb = setup_sb()
        concepts, preds, item_ids = self.get_concepts_and_preds(sb)

        # Create sketch
        sketch = Sketch(
            sb,
            concepts=[c.id for c in concepts.values()],
            model_type=msb.ModelType.ManualLinear,
            output_type=msb.OutputType.Continuous,
        )
        manual_weights = {
            concepts["c_0"].id: -0.4,
            concepts["c_1"].id: 0.1,
            concepts["c_2"].id: 0.3,
        }
        sketch.train(dataset_id="d_0", manual_weights=manual_weights)
        df_out = sketch.visualize(dataset_id="d_0", is_styled=False)

        # Validate concept score values
        df_item_ids = df_out["msb_item_id"].tolist()
        for concept_letter, concept in concepts.items():
            concept_id = concept.id
            df_preds = df_out[concept_id].tolist()

            observed = {
                item_id: pred_val for item_id, pred_val in zip(df_item_ids, df_preds)
            }
            expected = preds[concept_letter]
            assert (
                observed == expected
            ), f"Concept {concept_letter} predictions did not match expected values"

        # Validate sketch prediction
        expected_arrs = [
            (manual_weights[c_id] * np.array(list(preds[c_id].values())))
            for c_id in sb.concepts.keys()
        ]
        expected_vals = np.sum(expected_arrs, axis=0)

        sketch_pred_col = "sketch_pred"
        expected = {item_ids[i]: expected_vals[i] for i in range(N_ROWS)}
        df_sketch_pred = df_out[sketch_pred_col].tolist()
        observed_sketch = {
            item_id: pred_val for item_id, pred_val in zip(df_item_ids, df_sketch_pred)
        }
        assert (
            observed_sketch == expected
        ), f"Sketch predictions did not match expected values"

        # Validate ground truth
        gt_col = "overall_rating_score"
        expected_vals = sb.datasets["d_0"].df[gt_col]
        expected = {item_ids[i]: expected_vals[i] for i in range(N_ROWS)}
        df_gt = df_out[gt_col].tolist()
        observed_gt = {item_id: gt for item_id, gt in zip(df_item_ids, df_gt)}
        assert (
            observed_gt == expected
        ), f"Ground truth values did not match expected values"

        # Validate diff values
        diff_col = sketch._get_diff_col(sketch_pred_col, gt_col)
        expected = {
            item_ids[i]: (observed_sketch[item_ids[i]] - observed_gt[item_ids[i]])
            for i in range(N_ROWS)
        }
        df_diff = df_out[diff_col].tolist()
        observed_diff = {item_id: diff for item_id, diff in zip(df_item_ids, df_diff)}
        assert observed_diff == expected, f"Diff values did not match expected values"
