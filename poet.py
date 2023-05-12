from model import get_gen_model, gumbel_softmax
import tensorflow as tf
import numpy as np
from training import train
from preprocess import get_data
import re

EPOCHS = 0 

class Poet:

    def __init__(self):
        self.train, self.vocab = get_data('data/SmallerData.txt')
        self.embed = 128
        self.hidden_unit = 512
        # self.model_path = './models/love-letter-generator-model.h5'

        self.max_poem_length = 100
        # n, for multinomial distribution (hyperparemter that represents num outcomes)
        self.n = 3

    def predict_next_word(self, existing_poem):
        existing_poem_seq = []
        for word in existing_poem.split():
            if word in self.vocab:
                # remove punctuation and whitespace
                w = re.sub("[^\w\s]+", "", word)
                existing_poem_seq.append(self.vocab[w])
            else:
                existing_poem_seq.append(self.vocab['UNK'])
        # Pad the sequence to a fixed length (if necessary)
        existing_poem_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [existing_poem_seq], maxlen=self.max_poem_length, padding='post'
        )
        # gumbel + multinomial for training randomness 
        gumbel_noise = self.generator.predict(existing_poem_seq)
        gumbel_noise = gumbel_noise.ravel()
        gumbel_noise /= np.sum(gumbel_noise)
        random_sample = np.random.multinomial(self.n, gumbel_noise)
        max_prob_index = random_sample.argmax()
        # print(max_prob_index, len(self.vocab))
        integer_to_word = dict((value, key) for key, value in self.vocab.items())

        next_word = integer_to_word[max_prob_index]
        
        return next_word
    #minimize number of files.
    def generate(self, load):
        self.generator = None
        total_g = 0
        total_d = 0
        if not load:
            for epoch_id in range(EPOCHS):
                curr_g, curr_d, gen, dim = train(self.train, self.vocab)
                total_g = total_g + curr_g
                total_d = total_d + curr_d
                print('Epoch ', epoch_id, ': generator loss: ', curr_g)
                print('Epoch ', epoch_id, ': discriminator loss: ', curr_d)
    
            self.generator = gen
            gen.save('saved_model/my_gen')
            dim.save('saved_model/my_dim')
        else:
            self.generator = tf.keras.models.load_model('saved_model/my_gen')
            gen = self.generator
        return gen, total_g, total_d

    def create_poem(self, name, characteristic): # take in characteristic -- how would it generate words based on it
        first_sentence = name + " is a very " + characteristic + " person."
        poem = first_sentence
        while (len(poem) < self.max_poem_length):
            next_word = self.predict_next_word(poem)
            print(next_word, end='')
            poem = poem + " " + next_word
        return poem 

poet = Poet()
gen, total_g, total_d = poet.generate(False)
poem = poet.create_poem("Tabitha", "clumsy")
print(poem)
# average loss per epoch, where loss is the average loss per batch
print('Average G loss over all epoch: ', total_g/EPOCHS)
print('Average D loss over all epoch: ', total_d/EPOCHS)