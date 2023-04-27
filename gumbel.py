from typing import Any, Dict, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.layers import Layer

import nalp.utils.constants as c


def gumbel_distribution(input_shape: Tuple[int, ...]) -> tf.Tensor:
    """Samples a tensor from a Gumbel distribution.
    Args:
        input_shape: Shape of tensor to be sampled.
    Returns:
        (tf.Tensor): An input_shape tensor sampled from a Gumbel distribution.
    """

    uniform_dist = tf.random.uniform(input_shape, 0, 1)
    gumbel_dist = -1 * tf.math.log(
        -1 * tf.math.log(uniform_dist + c.EPSILON) + c.EPSILON
    )

    return gumbel_dist


class GumbelSoftmax(Layer):
    """A GumbelSoftmax class is the one in charge of a Gumbel-Softmax layer implementation.
    References:
        E. Jang, S. Gu, B. Poole. Categorical reparameterization with gumbel-softmax.
        Preprint arXiv:1611.01144 (2016).
    """

    def __init__(self, axis: Optional[int] = -1, **kwargs) -> None:
        """Initialization method.
        Args:
            axis: Axis to perform the softmax operation.
        """

        super(GumbelSoftmax, self).__init__(**kwargs)

        self.axis = axis

    def call(self, inputs: tf.Tensor, tau: float) -> Tuple[tf.Tensor, tf.Tensor]:
        """Method that holds vital information whenever this class is called.
        Args:
            x: A tensorflow's tensor holding input data.
            tau: Gumbel-Softmax temperature parameter.
        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): Gumbel-Softmax output and its argmax token.
        """

        x = inputs + gumbel_distribution(tf.shape(inputs))
        x = tf.nn.softmax(x / tau, self.axis)

        y = tf.stop_gradient(tf.argmax(x, self.axis, tf.int32))

        return x, y

    def get_config(self) -> Dict[str, Any]:
        """Gets the configuration of the layer for further serialization.
        Returns:
            (Dict[str, Any]): Configuration dictionary.
        """

        config = {"axis": self.axis}
        base_config = super(GumbelSoftmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))