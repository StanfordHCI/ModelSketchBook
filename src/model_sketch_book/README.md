# model_sketch_book — getting started

## Imports
```
import model_sketch_book as msb
from msb_enums import InputType
```

## Set up the sketchbook
```
sb = msb.create_model_sketchbook(
    goal='Example sketchbook goal here',
    schema={
        # Specify your data schema here
        "picture_url": InputType.Image,
        "neighborhood_overview": InputType.Text,
        "description": InputType.Text,
        "overall_rating": InputType.GroundTruth,  # required
    },
    credentials={
        "organization": "org-INSERT",
        "api_key": "sk-INSERT"
    }
)
```

## Add your dataset(s)
Then, add your dataset from a Pandas dataframe (recommended: 40-50 rows). If the dataframe contains images, this step will involve caching the images to speed up processing later.
```
listings = pd.read_csv("../data/michelle_data/listings.csv") # insert logic to load your dataframe

sb.add_dataset(
    df=listings,
    default=True,  # if dataset should be used by default, otherwise omit argument
)
```

## Create concepts
You can then go ahead and create image and text concepts with the following function. This function will display widgets to specify your concept term, input field, and output type.
```
msb.create_concept_model(sb)
```

<img src="../../docs/media/23_02_21_create_concept_model.png" alt="create_concept_model widgets" width="35%">

## Tune concepts (optional)
You may optionally tune your existing concepts by binarizing them at a threshold, normalizing the values, or calibrating them between specified values. This function will display widgets to select an existing concept, a tuning method, and tuning-related parameters.
```
msb.tune_concept(sb)
```

<img src="../../docs/media/23_02_21_tune_concept.png" alt="tune_concept widgets" width="50%">

## Create sketches
Then, you can combine concepts together into sketches with the following function. This function will display widgets to select concepts and an aggregator (only Linear Regression is supported currently).
```
msb.create_sketch_model(sb)
```

<img src="../../docs/media/23_02_21_create_sketch_model.png" alt="create_sketch_model widgets" width="50%">

## Other concept types
### Logical concepts (AND, OR)
Logical concepts can be applied to any number of existing binary concepts. This function will display widgets to select the concepts and the logical operator (AND or OR) to apply to those concept scores.
```
msb.create_logical_concept_model(sb)
```

<img src="../../docs/media/23_02_21_create_logical_concept_model.png" alt="create_sketch_model widgets" width="50%">

### Keyword concepts
Keyword concepts can be applied to any text-based input fields. The text of each example will be compared to the specified list of keywords, and if any of the keywords appear in the example, the example will be given a positive (True) label. This function will display widgets to select the input field and the comma-separated list of keywords, which can be treated in a case-sensitive or case-insensitive manner.
```
msb.create_keyword_concept_model(sb)
```

<img src="../../docs/media/23_02_21_create_keyword_concept_model.png" alt="create_sketch_model widgets" width="35%">

## Helper functions
### Take notes
```
msb.take_note(sb)
```

### Concept helper functions
#### View existing concepts
```
msb.show_concepts(sb)
```

#### Get concept term suggestions
```
msb.get_similar_concepts("your concept term")
```

#### Compare concepts to ground truth labels
```
msb.compare_concepts_to_gt(sb)
```

#### Compare concepts to each other
```
msb.compare_two_concepts(sb)
```

### Sketch helper functions
#### View existing sketches
```
msb.show_sketches(sb)
```

#### Test an existing sketch on a dataset
```
msb.test_sketch(sb)
```

#### Compare sketch performance
```
msb.compare_sketches(sb)
```
