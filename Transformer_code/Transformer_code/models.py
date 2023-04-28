from tensorflow.keras.layers import (
    Input,
    GlobalAvgPool1D,
    Dense,
    Bidirectional,
    GRU,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.losses import mae

from transformer import Encoder
import tensorflow as tf


def transformer_classifier(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=256,
    maximum_position_encoding=2048,
    n_classes=16,
):
    inp = Input((None, d_model))

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3,
    )
    # print("inp",inp.shape)
    x = encoder(inp)
    # print("x",x.shape)

    x = Dropout(0.2)(x)

    x = GlobalAvgPool1D()(x)

    x = Dense(4 * n_classes, activation="selu")(x)

    out = Dense(n_classes, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.00100)

    model.compile(
        optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()

    return model


def transformer_pretrain(
    num_layers=4, d_model=128, num_heads=8, dff=256, maximum_position_encoding=2048,
):
    inp = Input((None, d_model))

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=maximum_position_encoding,
        rate=0.3,
    )

    x = encoder(inp)

    out = Dense(d_model, activation="linear", name="out_pretraining")(x)

    model = Model(inputs=inp, outputs=out)

    opt = Adam(0.0001)

    model.compile(optimizer=opt, loss=mae)

    model.summary()

    return model




if __name__ == "__main__":
    model1 = transformer_classifier()

    # model2 = rnn_classifier()
