import tensorflow as tf
import numpy as np
from preprocess import get_data
from model import get_gen_model, gumbel_softmax, get_disc_model, g_loss, d_loss

EPOCHS = 1
BATCH_SIZE = 30
HIDDEN_UNIT = 512
UNITS = 200
EMBED = 128

def train(train_data, vocab_dict):
    total_loss = 0
    g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    
    num_samples = len(train_data)
    num_batches = tf.math.ceil(num_samples/BATCH_SIZE)
    batches = np.array_split(train_data, num_batches) # creating list of batches based on num batches

    generator = get_gen_model(batch_sz=1, encoding_dimension = [len(vocab_dict), EMBED], hidden_unit = HIDDEN_UNIT, optimizer = g_optimizer)
    discriminator = get_disc_model(UNITS, d_optimizer)

    for batch in batches:
        # calculate and apply generator gradients
        with tf.GradientTape() as tape:
            #batch = batch.reshape((30, 1)) 
            batch = batch.reshape((batch.shape[0], 1)) 
            x_fake = gumbel_softmax(generator(batch)) #(batch size, window size, vocab size)
            print('generator value for x_fake: ', x_fake)
            d_fake = discriminator(x_fake)
            print("generator value for d_fake: ", d_fake)
            loss = g_loss(d_fake)
        grads = tape.gradient(loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        print('finished generator loss section')

        #calculate and apply discriminator gradients, get the real loss too
        with tf.GradientTape() as tape:
            x_real = generator(batch)
            x_fake = gumbel_softmax(generator(batch)) #(batch size, window size, vocab size)
            print('discriminator value for x_fake: ', x_fake)
            d_fake = discriminator(x_fake)
            print("discriminator value for d_fake: ", d_fake)
            d_real = discriminator(x_real)
            loss = d_loss(d_fake, d_real)
            total_loss += loss
        grads = tape.gradient(loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        print('finished discriminator loss section')


train_data, vocab_dict = get_data("data/Poetry.txt")
for epoch_id in range(EPOCHS):
    total_loss = train(train_data, vocab_dict)
    print("Training Epoch: ", epoch_id, " and Loss: ", total_loss)
    #print("Training Epoch: ", epoch_id, " and Loss: ", total_loss/len(train_data))


