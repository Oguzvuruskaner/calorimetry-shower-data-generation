from typing import Union

import tensorflow as tf


class FeatureMatching(tf.keras.layers.Layer):

    # Used after an activation function.
    # Works coordinated with the other FeatureMatching layers in same model.
    # In training process it randomly cuts forward propagation and returns backward propagation.

    _FEATURE_MATHCING_LAYERS = {

    }

    _MODEL_ID = 0

    def __init__(self, model_id: Union[int, None] = None, **kwargs):

        super().__init__(**kwargs)
        if model_id is not None:
            FeatureMatching._MODEL_ID = model_id