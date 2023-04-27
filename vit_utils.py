# This script is adapted from https://github.com/faustomorales/vit-keras
# The implementation of TransformerBlock is refined so that the tensor ordering is consistent
# with operation ordering.
import tensorflow as tf
import tensorflow_addons as tfa
import typing
import warnings
import typing_extensions as tx
import typing
import warnings
import numpy as np
import scipy as sp


# @tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
# @tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block.
       This implementation makes the order of trainables 
       consistent with the execution order of operations.
    """

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):

        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )

        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )

        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        
    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


######################################################################

def apply_embedding_weights(target_layer, source_weights, num_x_patches, num_y_patches):
    """Apply embedding weights to a target layer.

    Args:
        target_layer: The target layer to which weights will
            be applied.
        source_weights: The source weights, which will be
            resized as necessary.
        num_x_patches: Number of patches in width of image.
        num_y_patches: Number of patches in height of image.
    """
    expected_shape = target_layer.weights[0].shape
    if expected_shape != source_weights.shape:
        token, grid = source_weights[0, :1], source_weights[0, 1:]
        sin = int(np.sqrt(grid.shape[0]))
        sout_x = num_x_patches
        sout_y = num_y_patches
        warnings.warn(
            "Resizing position embeddings from " f"{sin}, {sin} to {sout_x}, {sout_y}",
            UserWarning,
        )
        zoom = (sout_y / sin, sout_x / sin, 1)
        grid = sp.ndimage.zoom(grid.reshape(sin, sin, -1), zoom, order=1).reshape(
            sout_x * sout_y, -1
        )
        source_weights = np.concatenate([token, grid], axis=0)[np.newaxis]
    target_layer.set_weights([source_weights])


def load_weights_numpy(
    model, params_path, pretrained_top, num_x_patches, num_y_patches
):
    """Load weights saved using Flax as a numpy array.

    Args:
        model: A Keras model to load the weights into.
        params_path: Filepath to a numpy archive.
        pretrained_top: Whether to load the top layer weights.
        num_x_patches: Number of patches in width of image.
        num_y_patches: Number of patches in height of image.
    """
    params_dict = np.load(
        params_path, allow_pickle=False
    )  # pylint: disable=unexpected-keyword-arg
    source_keys = list(params_dict.keys())
    pre_logits = any(l.name == "pre_logits" for l in model.layers)
    source_keys_used = []
    n_transformers = len(
        set(
            "/".join(k.split("/")[:2])
            for k in source_keys
            if k.startswith("Transformer/encoderblock_")
        )
    )
    n_transformers_out = sum(
        l.name.startswith("Transformer/encoderblock_") for l in model.layers
    )
    assert n_transformers == n_transformers_out, (
        f"Wrong number of transformers ("
        f"{n_transformers_out} in model vs. {n_transformers} in weights)."
    )

    matches = []
    for tidx in range(n_transformers):
        encoder = model.get_layer(f"Transformer/encoderblock_{tidx}")
        source_prefix = f"Transformer/encoderblock_{tidx}"
        matches.extend(
            [
                {
                    "layer": layer,
                    "keys": [
                        f"{source_prefix}/{norm}/{name}" for name in ["scale", "bias"]
                    ],
                }
                for norm, layer in [
                    ("LayerNorm_0", encoder.layernorm1),
                    ("LayerNorm_2", encoder.layernorm2),
                ]
            ]
            + [
                {
                    "layer": encoder.mlpblock.get_layer(
                        f"{source_prefix}/Dense_{mlpdense}"
                    ),
                    "keys": [
                        f"{source_prefix}/MlpBlock_3/Dense_{mlpdense}/{name}"
                        for name in ["kernel", "bias"]
                    ],
                }
                for mlpdense in [0, 1]
            ]
            + [
                {
                    "layer": layer,
                    "keys": [
                        f"{source_prefix}/MultiHeadDotProductAttention_1/{attvar}/{name}"
                        for name in ["kernel", "bias"]
                    ],
                    "reshape": True,
                }
                for attvar, layer in [
                    ("query", encoder.att.query_dense),
                    ("key", encoder.att.key_dense),
                    ("value", encoder.att.value_dense),
                    ("out", encoder.att.combine_heads),
                ]
            ]
        )
    for layer_name in ["embedding", "head", "pre_logits"]:
        if layer_name == "head" and not pretrained_top:
            source_keys_used.extend(["head/kernel", "head/bias"])
            continue
        if layer_name == "pre_logits" and not pre_logits:
            continue
        matches.append(
            {
                "layer": model.get_layer(layer_name),
                "keys": [f"{layer_name}/{name}" for name in ["kernel", "bias"]],
            }
        )
    matches.append({"layer": model.get_layer("class_token"), "keys": ["cls"]})
    matches.append(
        {
            "layer": model.get_layer("Transformer/encoder_norm"),
            "keys": [f"Transformer/encoder_norm/{name}" for name in ["scale", "bias"]],
        }
    )
    apply_embedding_weights(
        target_layer=model.get_layer("Transformer/posembed_input"),
        source_weights=params_dict["Transformer/posembed_input/pos_embedding"],
        num_x_patches=num_x_patches,
        num_y_patches=num_y_patches,
    )
    source_keys_used.append("Transformer/posembed_input/pos_embedding")
    for match in matches:
        source_keys_used.extend(match["keys"])
        source_weights = [params_dict[k] for k in match["keys"]]
        if match.get("reshape", False):
            source_weights = [
                source.reshape(expected.shape)
                for source, expected in zip(
                    source_weights, match["layer"].get_weights()
                )
            ]
        match["layer"].set_weights(source_weights)
    unused = set(source_keys).difference(source_keys_used)
    if unused:
        warnings.warn(f"Did not use the following weights: {unused}", UserWarning)
    target_keys_set = len(source_keys_used)
    target_keys_all = len(model.weights)
    if target_keys_set < target_keys_all:
        warnings.warn(
            f"Only set {target_keys_set} of {target_keys_all} weights.", UserWarning
        )

######################################################################

ConfigDict = tx.TypedDict(
    "ConfigDict",
    {
        "dropout": float,
        "mlp_dim": int,
        "num_heads": int,
        "num_layers": int,
        "hidden_size": int,
    },
)

CONFIG_B: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L: ConfigDict = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 1024,
}

BASE_URL = "https://github.com/faustomorales/vit-keras/releases/download/dl"
WEIGHTS = {"imagenet21k": 21_843, "imagenet21k+imagenet2012": 1_000}
SIZES = {"B_16", "B_32", "L_16", "L_32"}

ImageSizeArg = typing.Union[typing.Tuple[int, int], int]


def preprocess_inputs(X):
    """Preprocess images"""
    return tf.keras.applications.imagenet_utils.preprocess_input(
        X, data_format=None, mode="tf"
    )


def interpret_image_size(image_size_arg: ImageSizeArg) -> typing.Tuple[int, int]:
    """Process the image_size argument whether a tuple or int."""
    if isinstance(image_size_arg, int):
        return (image_size_arg, image_size_arg)
    if (
        isinstance(image_size_arg, tuple)
        and len(image_size_arg) == 2
        and all(map(lambda v: isinstance(v, int), image_size_arg))
    ):
        return image_size_arg
    raise ValueError(
        f"The image_size argument must be a tuple of 2 integers or a single integer. Received: {image_size_arg}"
    )


def build_model(
    image_size: ImageSizeArg,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
    image_size_tuple = interpret_image_size(image_size)
    assert (image_size_tuple[0] % patch_size == 0) and (
        image_size_tuple[1] % patch_size == 0
    ), "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size_tuple[0], image_size_tuple[1], 3))
    y = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(x)
    y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = ClassToken(name="class_token")(y)
    y = AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh"
        )(y)
    if include_top:
        y = tf.keras.layers.Dense(classes, name="head", activation=activation)(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


def validate_pretrained_top(
    include_top: bool, pretrained: bool, classes: int, weights: str
):
    """Validate that the pretrained weight configuration makes sense."""
    assert weights in WEIGHTS, f"Unexpected weights: {weights}."
    expected_classes = WEIGHTS[weights]
    if classes != expected_classes:
        warnings.warn(
            f"Can only use pretrained_top with {weights} if classes = {expected_classes}. Setting manually.",
            UserWarning,
        )
    assert include_top, "Can only use pretrained_top with include_top."
    assert pretrained, "Can only use pretrained_top with pretrained."
    return expected_classes


def load_pretrained(
    size: str,
    weights: str,
    pretrained_top: bool,
    model: tf.keras.models.Model,
    image_size: ImageSizeArg,
    patch_size: int,
):
    """Load model weights for a known configuration."""
    image_size_tuple = interpret_image_size(image_size)
    fname = f"ViT-{size}_{weights}.npz"
    origin = f"{BASE_URL}/{fname}"
    local_filepath = tf.keras.utils.get_file(fname, origin, cache_subdir="weights")
    load_weights_numpy(
        model=model,
        params_path=local_filepath,
        pretrained_top=pretrained_top,
        num_x_patches=image_size_tuple[1] // patch_size,
        num_y_patches=image_size_tuple[0] // patch_size,
    )


def vit_b16(
    image_size: ImageSizeArg = (224, 224),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B16. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        patch_size=16,
        image_size=image_size,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )

    if pretrained:
        load_pretrained(
            size="B_16",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            image_size=image_size,
            patch_size=16,
        )
    return model


def vit_b32(
    image_size: ImageSizeArg = (224, 224),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-B32. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_B,
        name="vit-b32",
        patch_size=32,
        image_size=image_size,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    if pretrained:
        load_pretrained(
            size="B_32",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            patch_size=32,
            image_size=image_size,
        )
    return model


def vit_l16(
    image_size: ImageSizeArg = (384, 384),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-L16. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_L,
        patch_size=16,
        name="vit-l16",
        image_size=image_size,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=1024 if weights == "imagenet21k" else None,
    )
    if pretrained:
        load_pretrained(
            size="L_16",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            patch_size=16,
            image_size=image_size,
        )
    return model


def vit_l32(
    image_size: ImageSizeArg = (384, 384),
    classes=1000,
    activation="linear",
    include_top=True,
    pretrained=True,
    pretrained_top=True,
    weights="imagenet21k+imagenet2012",
):
    """Build ViT-L32. All arguments passed to build_model."""
    if pretrained_top:
        classes = validate_pretrained_top(
            include_top=include_top,
            pretrained=pretrained,
            classes=classes,
            weights=weights,
        )
    model = build_model(
        **CONFIG_L,
        patch_size=32,
        name="vit-l32",
        image_size=image_size,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=1024 if weights == "imagenet21k" else None,
    )
    if pretrained:
        load_pretrained(
            size="L_32",
            weights=weights,
            model=model,
            pretrained_top=pretrained_top,
            patch_size=32,
            image_size=image_size,
        )
    return model

