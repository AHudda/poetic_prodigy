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
    # print('returned model')
    
    return model


cc_func = tf.keras.backend.binary_crossentropy
acc_func = tf.keras.metrics.binary_accuracy
# g_l = tf.Variable(tf.random.truncated_normal([29, 1]))
# d_l = tf.Variable(tf.random.truncated_normal([29, 1]))
# d_r = tf.Variable(tf.random.truncated_normal([29, 1]))

def g_loss(d_fake:tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(cc_func(tf.ones_like(d_fake), d_fake))

def d_loss(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    d_fake_loss = tf.reduce_mean(cc_func(tf.zeros_like(d_fake), d_fake))
    d_real_loss = tf.reduce_mean(cc_func(tf.ones_like(d_real), d_real))
    return d_fake_loss + d_real_loss

def d_acc_fake(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    ideal=tf.zeros_like(d_fake) #using fake data, how many times was it fake and discriminator could tell it was fake #prediction that discriminator says the fake thing is fake
    return acc_func(ideal, d_fake) 

def d_acc_real(d_fake:tf.Tensor, d_real:tf.Tensor)  -> tf.Tensor:
    ideal=tf.ones_like(d_real) #acc real we use 1s like and real data
    return acc_func(ideal, d_real)

def g_acc(d_fake:tf.Tensor, d_real:tf.Tensor) -> tf.Tensor:
    ideal=tf.ones_like(d_fake) #prob it's able to fool discriminator. zeros-like - how good discriminator is true
    return acc_func(ideal, d_fake) #how many times is it fooling discriminator

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

