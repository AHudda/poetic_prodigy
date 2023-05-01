import tensorflow as tf
import gumbel as gumbel

def build_generator(batch_sz, encoding_dimension, hidden_unit, optimizer):
    model = tf.keras.layers.Sequential([
        tf.keras.layers.Embedding(input_dim = encoding_dimension[0], output_dim = encoding_dimension[1],
                                  batch_input_shape = [batch_sz, None]),
        tf.keras.layers.LSTM(units = hidden_unit, return_sequences=True, recurrent_initializer='glorot_uniform')       
    ])
    model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True),
                  optimizer = optimizer)

    return model


# takes in the LSTM output with shape (batch size, window size, vocab size)
def gumbel_softmax(input): 
    return gumbel(input, 1.0) # temp is a hyperparameter



#def build_critic():

