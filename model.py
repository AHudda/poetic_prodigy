import tensorflow as tf
from gumbel import GumbelSoftmax
import tensorflow_probability as tfp


def get_gen_model(batch_sz, encoding_dimension, hidden_unit, optimizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim = encoding_dimension[0], output_dim = encoding_dimension[1]), #, input_shape=(None,)
        tf.keras.layers.LSTM(units = hidden_unit, return_sequences=False, recurrent_initializer='glorot_uniform', activation = 'sigmoid')       
    ])
    
    model.compile(optimizer=optimizer, loss=g_loss)

    return model


# takes in the LSTM output with shape (batch size, window size, vocab size)
def gumbel_softmax(input): 
    print('in gumbel')
    x = tfp.distributions.RelaxedOneHotCategorical(temperature = 1, probs = input)
    print('this is the output of gumbel_softmax: ', x.sample())
    return x

    # GumbelSoftmax(input)
    #tfp.distributions.RelaxedOneHotCategorical(temperature = 1, probs = input)


def get_disc_model(units, optimizer):
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
    print('returned model')
    
    return model

# logits_real: Tensor, shape [batch_size, 1], output of discriminator for each real image
# logits_fake: Tensor, shape[batch_size, 1], output of discriminator for each fake image
scc_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

def g_loss(d_fake:tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(scc_func(tf.ones_like(d_fake), d_fake))

def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    d_fake_loss = scc_func(tf.zeros_like(d_fake), d_fake)
    d_real_loss = scc_func(tf.ones_like(d_real), d_real)
    return tf.reduce_mean(d_fake_loss + d_real_loss)

