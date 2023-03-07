# RESULT


# Class definition of a result
# - id: unique identifier string
# - preds: dict mapping item id to a prediction
class Result:
    def __init__(
        self,
        id,
        preds,
    ):
        self.id = id
        self.preds = preds
