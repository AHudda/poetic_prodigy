import tensorflow as tf

def build_model(batch_sz, encoding_dimension, hidden_unit, optimizer):
    model = tf.keras.layers.Sequential([
        tf.keras.layers.Embedding(input_dim = encoding_dimension[0], output_dim = encoding_dimension[1],
                                  batch_input_shape = [batch_sz, None]),
        tf.keras.layers.LSTM(units = hidden_unit, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.tf_agents.distributions.gumbel_softmax.GumbelSoftmax(1.0, )                          
    ])