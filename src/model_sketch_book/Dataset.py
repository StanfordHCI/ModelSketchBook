# DATASET

import pandas as pd


# Class definition of a dataset
# Parameters:
# - id: unique identifier string
# - df: dataframe to associate with the dataset
# - ground_truth: name of column with ground truth labels, if applicable
# Attributes:
# - labeled: whether or not this dataset has ground truth labels
# - cached_images: dictionary from input field to dict of {item_id: Image} for fields that are images; caching happens in SketchBook's add_dataset() function
# - cached_images_html: dictionary from input field to dict of {item_id: image HTML string} for display in the visualization dataframe; caching happens in SketchBook's add_dataset() function
class Dataset:
    def __init__(
        self,
        id: str,
        df: pd.DataFrame,
        ground_truth: str = None,
    ):
        self.id = id
        # Add a unique item identifier column
        df_new = df.copy()
        df_new.insert(0, "msb_item_id", [i for i in range(len(df))])
        self.df = df_new
        self.ground_truth = ground_truth
        self.labeled = True if ground_truth else False

        # Cached images
        self.cached_images = {}
        self.cached_images_html = {}

        # Cached concept results
        # Key = (concept_term, input_field, output_type)
        # Value = Result object
        # See get_concept_key() function in Concept class, which creates this key
        self.cached_concept_res = {}

    def get_item_ids(self):
        return self.df["msb_item_id"].tolist()
