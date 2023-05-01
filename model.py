import tensorflow as tf
from gumbel import call
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
    x = call(input, 1.0)
    return x[0]

    #tfp.distributions.RelaxedOneHotCategorical(temperature = 1, probs = input)
    #print('this is the output of gumbel_softmax: ', x.sample())

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


cc_func = tf.keras.losses.CategoricalCrossentropy()

def g_loss(d_fake:tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(cc_func(tf.ones_like(d_fake), d_fake))

def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    d_fake_loss = cc_func(tf.zeros_like(d_fake), d_fake)
    d_real_loss = cc_func(tf.ones_like(d_real), d_real)
    return tf.reduce_mean(d_fake_loss + d_real_loss)

