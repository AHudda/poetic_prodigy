from model import get_gen_model, gumbel_softmax, g_loss
import tensorflow as tf
import numpy as np
from training import train
from preprocess import get_data
import re

class Poet:

    def __init__(self):
        self.train, self.vocab = get_data('data/SmallerData.txt')
        self.embed = 128
        self.hidden_unit = 512
        self.max_poem_length = 100
        # self.model_path = './models/love-letter-generator-model.h5'
        
        # Hyperparameter n: represents number of outcomes for multinomial distribution
        self.n = 300


    def predict_next_word(self, existing_poem):
        """Predicts the next word in the poem.
        Args:
            existing_poem: Words in the existing poem.
        Returns:
            (String): Next word to be added to poem.
        """
        existing_poem_seq = []
        for word in existing_poem.split():
            if word in self.vocab:
                # Remove punctuation and whitespace
                w = re.sub("[^\w\s]+", "", word)
                existing_poem_seq.append(self.vocab[w])
            else:
                existing_poem_seq.append(self.vocab['UNK'])
        
        # Pad the sequence to a fixed length (if necessary)
        existing_poem_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [existing_poem_seq], maxlen=self.max_poem_length, padding='post'
        )
       
        # Gumbel & multinomial for training randomness
        gumbel_noise = self.generator.predict(existing_poem_seq)
        gumbel_noise = gumbel_noise.ravel()
        gumbel_noise /= np.sum(gumbel_noise)
        # Added the following lines (defining a) for Windows compatibility
        a = np.asarray(gumbel_noise).astype('float64')
        a = a / np.sum(a)
        random_sample = np.random.multinomial(self.n, a)
        max_prob_index = random_sample.argmax()
        integer_to_word = dict((value, key) for key, value in self.vocab.items())
        next_word = integer_to_word[max_prob_index]

        return next_word
   
    
    def generate(self, load, epochs):
        """Trains or loads the GAN model.
        Args:
            load: Boolean value indicating whether to train or load model.
            epochs: Number of epochs to train the model.
        Returns:
            (tf.keras.Sequential): Generator model.
            (float): total generator loss
            (float): total discriminator loss
        """
        self.generator = None
        total_g = 0
        total_d = 0
        
        if not load:
            for epoch_id in range(epochs):
                curr_g, curr_d, gen, dim = train(self.train, self.vocab)
                total_g = total_g + curr_g
                total_d = total_d + curr_d
                print('Epoch ', epoch_id, ': generator loss: ', curr_g)
                print('Epoch ', epoch_id, ': discriminator loss: ', curr_d)
            self.generator = gen
            gen.save('saved_model/my_gen')
            dim.save('saved_model/my_dim')
            
            # Average loss per epoch, where loss is the average loss per batch
            print('Average G loss over all epoch: ', total_g/epochs)
            print('Average D loss over all epoch: ', total_d/epochs)
        
        else:
            self.generator = tf.keras.models.load_model('saved_model/my_gen', custom_objects={'g_loss':g_loss})
            gen = self.generator
        return gen, total_g, total_d


    def create_poem(self, name, characteristic):
        """Takes in the prompt and creates the poem.
        Args:
            name: user input name.
            characteristic: user input characteristic.
        Returns:
            (String): poem.
        """
        first_sentence = name + " is a very " + characteristic + " person."
        poem = first_sentence
        while (len(poem) < self.max_poem_length):
            next_word = self.predict_next_word(poem)
            print(next_word, end='')
            poem = poem + " " + next_word
        return poem