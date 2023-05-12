import tensorflow as tf
import numpy as np
from model import get_gen_model, gumbel_softmax, get_disc_model, g_loss, d_loss, d_acc_fake, d_acc_real, g_acc

# Hyperparameters
BATCH_SIZE = 30
HIDDEN_UNIT = 512
UNITS = 200
EMBED = 128


def train(train_data, vocab_dict):
    """Predicts the next word in the poem.
    Args:
        train_data: Tokenized poem.
        vocab_dict: Dictionary mapping unique vocabulary word to unique numerical token
    Returns:
        (tf.Tensor): Average generator loss per batch.
        (tf.Tensor): Average discriminator loss per batch.
        (tf.keras.Sequential): Generator model.
        (tf.keras.Sequential): Discriminator model.
    """
    # Initialize cumulative losses
    total_g_loss = 0
    total_d_loss = 0

    # Initialize optimizers
    g_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
   
    # Batch train_data
    num_samples = len(train_data)
    num_batches = tf.math.ceil(num_samples/BATCH_SIZE)
    batches = np.array_split(train_data, num_batches)

    # Get generator and discriminator models
    generator = get_gen_model(batch_sz=1, encoding_dimension = [len(vocab_dict), EMBED], hidden_unit = HIDDEN_UNIT, optimizer = g_optimizer)
    discriminator = get_disc_model(UNITS, d_optimizer)

    for batch in batches:
        # Calculate and apply generator gradients
        with tf.GradientTape() as tape:
            batch = batch.reshape((batch.shape[0], 1))
            x_fake = gumbel_softmax(generator(batch))
            d_fake = discriminator(x_fake)
            loss = g_loss(d_fake)
            total_g_loss += loss
            # acc = g_acc(x_fake, None)
            # avg_gen_acc += acc
        grads = tape.gradient(loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # Calculate and apply discriminator gradients
        with tf.GradientTape() as tape:
            x_real = generator(batch)
            x_fake = gumbel_softmax(generator(batch))
            d_fake = discriminator(x_fake)
            d_real = discriminator(x_real)
            loss = d_loss(d_fake, d_real)
            total_d_loss += loss
            # f_acc = d_acc_fake(d_fake, None)
            # r_acc = d_acc_real(None, d_real)
            # avg_dis_acc += (f_acc *x_fake.shape[0] + r_acc*x_real.shape[0])/(x_fake.shape[0] + x_real.shape[0])
        grads = tape.gradient(loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    return total_g_loss / num_batches, total_d_loss / num_batches, generator, discriminator
