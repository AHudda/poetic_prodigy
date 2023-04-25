import tensorflow as tf
import numpy as np
import re
import pandas as pd

def get_data(file_name):
    """
    Helpful documentation
    """
    dataframe = pd.read_csv("data/PoetryFoundationData.csv")
    dataframe["Poem"].to_csv('data/Poetry.txt', sep="\n", index=False, header=False)

    train = []
    vocab = {}
    vocab_size = 0
    # file_name = "data/PoetryFoundationData.txt"
    print(file_name)
    with open(file_name, "r") as file:
        # file = file.lower()
        for line in file:
            line = re.sub("[^\w\s]", "", line)
            tokens = line.split()
            train.extend(tokens) #no argument also means white space
            #will feed elements of list into list ur sending
            for t in tokens: 
                #building up vocab
                if t not in vocab:
                    vocab[t] = vocab_size
                    vocab_size += 1
    print(train)
    train = list(map(lambda x: vocab[x], train))
    # print(train)
    return train, vocab
get_data("data/Poetry.txt")