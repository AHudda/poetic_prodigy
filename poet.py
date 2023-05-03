from model import get_gen_model, gumbel_softmax
import tensorflow as tf
import numpy as np
from training import train
from preprocess import get_data
import re

EPOCHS = 10 #change to 1 to save time?

class Poet:

    def __init__(self):
        self.train, self.vocab = get_data('data/SmallData.txt')
        # self.vocab_dict = vocab_dict
        self.embed = 128
        self.hidden_unit = 512
        # self.model_path = './models/love-letter-generator-model.h5'
        # wut

        self.max_poem_length = 100
        self.n = 3 # n, for multinomial distribution (hyperparemter that represents num outcomes)

    def predict_next_word(self, existing_poem): #have to get rid of periods
        existing_poem_seq = []
        for word in existing_poem.split():
            if word in self.vocab:
                w = re.sub("[^\w\s]+", "", word)
                existing_poem_seq.append(self.vocab[w])
            else:
                existing_poem_seq.append(self.vocab['UNK'])
        # Pad the sequence to a fixed length (if necessary)
        existing_poem_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [existing_poem_seq], maxlen=self.max_poem_length, padding='post'
        )
        # gumbel + multinomial for training randomness 
        gumbel_noise = self.generator.predict(existing_poem_seq) #self.generate()
        gumbel_noise = gumbel_noise.ravel()
        gumbel_noise /= np.sum(gumbel_noise)
        random_sample = np.random.multinomial(self.n, gumbel_noise)
        max_prob_index = random_sample.argmax()
        # print(max_prob_index, len(self.vocab))
        integer_to_word = dict((value, key) for key, value in self.vocab.items())

        next_word = integer_to_word[max_prob_index]
        
        return next_word
    #minimize number of files.
    def generate(self):
        for epoch_id in range(EPOCHS):
            total_g, total_d, gen, dim = train(self.train, self.vocab)
        # lstm_output = get_gen_model(batch_sz=1, encoding_dimension=[len(self.vocab_dict), self.embed], hidden_unit=self.hidden_unit, optimizer='adam')
        # random_distribution = gumbel_softmax(lstm_output)
        #model.load_weights()
        self.generator = gen
        gen.save('saved_model/my_gen')
        dim.save('saved_model/my_dim')
        return gen

    def create_poem(self, name, characteristic): # take in characteristic -- how would it generate words based on it
        first_sentence = name + " is a very " + characteristic + " person."
        poem = first_sentence
        while (len(poem) < self.max_poem_length):
            next_word = self.predict_next_word(poem)
            print(next_word, end='')
            poem = poem + next_word
            print(poem)
        
        return poem 

poet = Poet()
gen = poet.generate()
poem = poet.create_poem("rabbit", "crazed")
print(poem)