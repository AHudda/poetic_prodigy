# Poetic Prodigy: Personalized Poetry Generator
### Living Poets Society: Aanya Hudda (ahudda1), Katherine Mao (5kmao), Mason Zhang (mzhang150), Tabitha Lynn (tlynn1)
##

### How to run:
1. Clone the github repository
2. Activate csci1470 Anaconda environment
3. Type 'python repl.py' in the terminal
4. To terminate, press control c or command c

### Devpost Submission:
[Link here](https://devpost.com/software/poetic-prodigy-personalized-poetry-generator)

### Introduction: 
Inspired by the interdisciplinary relationship between art and technology, we developed a Generative Adversarial Network (GAN) with the goal of creating original poems. We hope to fuse the creativity of human expression and the power of deep learning in an interactive and accessible way. With this aim, we implemented [Creative GANS for generating poems, lyrics and metaphors](https://arxiv.org/abs/1909.09534) that uses GANs for creative text generation. Given a name and a characteristic, our GAN model will return a poem of 100 characters. 

### Data: 
We trained our model on a dataset consisting of 198 poems from [Poetry Foundation](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems). The dataset was in the form of a csv file with three columns: title, poem, poet. Using pandas, we read the poem column into a text file (Poetry.txt), which was used to preprocess and tokenize the poems. We then proceeded to use regexes to strip punctuation and white space from the data. After the poetry was processed, unique words were added to a dictionary mapping from the word to its number of occurrences, and a list of all the words was created and turned into a list of tokens. Both were passed into the aforementioned architecture to train the model. 

### Methodology:
As we originally articulated in our devpost, we drew heavily from the paper as well as [Coding Cupids – Love Letter Generator](https://devpost.com/software/coding-cupids-love-letter-generator) to build our model. We implemented the same architecture as the paper, but we used Tensorflow instead of Pytorch. Our generator consists of an embedding layer and three LSTM layers. In order to add noise to the generator output, we pass the generator output into gumbel softmax to produce x_fake. Our discriminator consists of three blocks of dense layer with Leaky RELU activation and batch normalization followed by a dense layer with sigmoid activation. We used binary cross entropy as the metric for both our generator and discriminator loss functions.

### Related Work:
- [Tokyo GAN](https://github.com/Machine-Learning-Tokyo/Poetry-GAN/blob/master/README.md)
