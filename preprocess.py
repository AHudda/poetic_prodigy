import tensorflow as tf
import numpy as np
import re
import pandas as pd


def get_data(file_name):
    """Gets and processes the data.
    Args:
        file_name: Name of file containing data.
    Returns:
        (List): List of tokenized poems
        (Dictionary): Dictionary mapping unique word to unique numerical token.
    """
    # Following lines load 800k line dataset:
    # dataframe = pd.read_csv("data/PoetryFoundationData.csv")
    # dataframe["Poem"].to_csv('data/Poetry.txt', sep="\n", index=False, header=False)

    train = []
    vocab = {}
    vocab_size = 0

    with open(file_name, "r", encoding="utf8") as file:
        for line in file:
            line = re.sub("[^\w\s]+", "", line) #added a plus
            tokens = line.split()
            train.extend(tokens) #no argument also means white space
           
            for t in tokens:
                if t not in vocab:
                    vocab[t] = vocab_size
                    vocab_size += 1
    
    vocab['UNK'] = vocab_size
    vocab_size += 1

    train = list(map(lambda x: vocab[x], train))
    return train, vocab
