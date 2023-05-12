from typing import Any, Dict, Optional, Tuple
import tensorflow as tf

EPSILON = 1e-20


def gumbel_distribution(input_shape: Tuple[int, ...]) -> tf.Tensor:
    """Samples a tensor from a Gumbel distribution.
    Args:
        input_shape: Shape of tensor to be sampled.
    Returns:
        (tf.Tensor): An input_shape tensor sampled from a Gumbel distribution.
    """
    uniform_dist = tf.random.uniform(input_shape, 0, 1)
    gumbel_dist = -1 * tf.math.log(
        -1 * tf.math.log(uniform_dist + EPSILON) + EPSILON
    )
    return gumbel_dist


def call(inputs: tf.Tensor, tau: float) -> Tuple[tf.Tensor, tf.Tensor]:
    """Method that applies gumbel softmax to inputs.
         Args:
            x: A tensorflow's tensor holding input data.
            tau: Gumbel-Softmax temperature parameter.
        Returns:
            (Tuple[tf.Tensor, tf.Tensor]): Gumbel-Softmax output and its argmax token.
    """
    x = inputs + gumbel_distribution(tf.shape(inputs))
    x = tf.nn.softmax(x / tau, -1)
    y = tf.stop_gradient(tf.argmax(x, -1, tf.int32))
    return x, y
