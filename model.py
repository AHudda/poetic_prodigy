import tensorflow as tf
from gumbel import call


def get_gen_model(batch_sz, encoding_dimension, hidden_unit, optimizer):
    """Creates the GAN generator.
    Args:
        batch_sz: Shape of batch input.
        encoding_dimension: Hyperparameter size of encoding table [vocab length, embedding size]
        hidden_unit: Dimensionality of LSTM output space
        optimizer: Adams optimizer for generator
    Returns:
        (tf.keras.Sequential): Generator model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim = encoding_dimension[0], output_dim = encoding_dimension[1]), #, input_shape=(None,)
        tf.keras.layers.LSTM(units = hidden_unit, return_sequences=True, recurrent_initializer='glorot_uniform', activation = 'sigmoid'),  
        tf.keras.layers.LSTM(units = hidden_unit, return_sequences=True, recurrent_initializer='glorot_uniform', activation = 'sigmoid'),
        tf.keras.layers.LSTM(units = hidden_unit, return_sequences=False, recurrent_initializer='glorot_uniform', activation = 'sigmoid')                  
    ])
    model.compile(optimizer=optimizer, loss=g_loss)
    return model


def gumbel_softmax(input):
    """Applies gumbel softmax to generator output.
    Args:
        input: Generator output of shape (batch size, window size, vocab size).
    Returns:
        (tf.Tensor): distorted probability distribution with noise
    """
    x = call(input, 1.0)
    return x[0]

    #tfp.distributions.RelaxedOneHotCategorical(temperature = 1, probs = input)


def get_disc_model(units, optimizer):
    """Creates the GAN discriminator.
    Args:
        units: Dimensionality of Dense layer output space
        optimizer: Adams optimizer for discriminator
    Returns:
        (tf.keras.Sequential): Discriminator model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units = units, activation = 'leaky_relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units = units, activation = 'leaky_relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units = units, activation = 'leaky_relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units = 1, activation = 'sigmoid')
    ])
    model.compile(optimizer = optimizer, loss = d_loss)   
    return model


cc_func = tf.keras.backend.binary_crossentropy
acc_func = tf.keras.metrics.binary_accuracy


def g_loss(d_fake:tf.Tensor) -> tf.Tensor:
    """Generator loss function.
    Args:
        d_fake: Discriminator output for x_fake.
    Returns:
        (tf.Tensor): Generator loss.
    """
    return tf.reduce_mean(cc_func(tf.ones_like(d_fake), d_fake))


def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    """Generator loss function.
    Args:
        d_fake: Discriminator output for x_fake.
        d_real: Discriminator output for x_real.
    Returns:
        (tf.Tensor): Discriminator loss.
    """
    d_fake_loss = tf.reduce_mean(cc_func(tf.zeros_like(d_fake), d_fake))
    d_real_loss = tf.reduce_mean(cc_func(tf.ones_like(d_real), d_real))
    return d_fake_loss + d_real_loss


def d_acc_fake(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    """Discriminator accuracy function for fake.
    Args:
        d_fake: Discriminator output for x_fake.
        d_real: Discriminator output for x_real.
    Returns:
        (tf.Tensor): Accuracy for d_fake.
    """
    # Using the fake data, how many times was it fake and the discriminator could tell it was fake
    # Prediction that discriminator says a fake thing is fake
    ideal=tf.zeros_like(d_fake) 
    return acc_func(ideal, d_fake)


def d_acc_real(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    """Discriminator accuracy function.
    Args:
        d_fake: Discriminator output for x_fake.
        d_real: Discriminator output for x_real.
    Returns:
        (tf.Tensor): Accuracy for d_real.
    """
    ideal=tf.ones_like(d_real)
    return acc_func(ideal, d_real)


def g_acc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    """Generator accuracy function.
    Args:
        d_fake: Discriminator output for x_fake.
        d_real: Discriminator output for x_real.
    Returns:
        (tf.Tensor): Accuracy for generator.
    """
    # Using the fake data, the probability that the generator is able to fool the discriminator
    ideal=tf.ones_like(d_fake)
    return acc_func(ideal, d_fake)


# Debugging Print Lines

# print("g: ", g_loss(g_l))
# print("d: ", d_loss(d_l, d_r))
# print(d_loss(tf.constant([0., 0., 0.]), tf.constant([1., 1., 1.])).numpy())
# print(d_loss(tf.constant([0., 0., 0.]), tf.constant([0., 0., 0.])).numpy())
# print(d_loss(tf.constant([1., 1., 1.]), tf.constant([1., 1., 1.])).numpy())
# print(d_loss(tf.constant([1., 1., 1.]), tf.constant([0., 0., 0.])).numpy())
# print()
# print(g_loss(tf.constant([1., 1., 1.]), None).numpy())
# print(g_loss(tf.constant([0., 0., 0.]), None).numpy())
# print()
# print(d_acc_fake(tf.constant([.1, .9, .9]), None).numpy())
# print(d_acc_real(None, tf.constant([.1, .9, .9])).numpy())
# print(g_acc(tf.constant([.1, .9, .9]), None).numpy())
