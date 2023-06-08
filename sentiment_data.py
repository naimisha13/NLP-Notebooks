# sentiment_data.py

from collections import Counter
from typing import List
from utils import *
import re
import numpy as np


class SentimentExample:
    """
    Wraps a sequence of word indices with a 0-1 label (0 = negative, 1 = positive)
    """
    def __init__(self, indexed_words, label: int):
        self.indexed_words = indexed_words
        self.label = label

    def __repr__(self):
        return repr(self.indexed_words) + "; label=" + repr(self.label)

    def get_indexed_words_reversed(self):
        return [self.indexed_words[len(self.indexed_words) - 1 - i] for i in range(0, len (self.indexed_words))]


def read_and_index_sentiment_examples(infile: str, indexer: Indexer, add_to_indexer=False, word_counter=None) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and indexes the sentence according
    to the vocabulary in indexer.
    :param infile: file to read
    :param indexer: Indexer containing the vocabulary
    :param add_to_indexer: If add_to_indexer is False, replaces unseen words with UNK, otherwise grows the indexer.
    :param word_counter: optionally keeps a tally of how many times each word is seen (mostly for logging purposes).
    :return: A list of SentimentExample objects read from the file
    """
    # f = open(infile, encoding='utf8')
    f = open(infile, encoding='iso8859')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if len(fields) < 2:         # To process imdb data properly
                fields = line.split(" ", 1)
            # Slightly more robust to reading bad output than int(fields[0])
            label = 0 if (("0" in fields[0]) or ("neg" in fields[0])) else 1
            sent = fields[1]
            tokenized_cleaned_sent = list(filter(lambda x: x != '', _clean_str(sent).strip().split(" ")))
            if word_counter is not None:
                for word in tokenized_cleaned_sent:
                    word_counter[word] += 1.0
            indexed_sent = [indexer.add_and_get_index(word) if indexer.contains(word) or add_to_indexer else indexer.index_of("UNK")
                 for word in tokenized_cleaned_sent]
            exs.append(SentimentExample(indexed_sent, label))
    f.close()
    return exs


def write_sentiment_examples(exs: List[SentimentExample], outfile: str, indexer):
    """
    Writes sentiment examples to an output file in the same format they are read in. Note that what gets written
    out is tokenized, so this will not exactly match the input file. However, this is fine from the standpoint of
    writing model output.
    :param exs: the list of SentimentExamples to write
    :param outfile: out path
    :return: None
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([indexer.get_object(idx) for idx in ex.indexed_words]) + "\n")
    o.close()


def _clean_str(string):
    """
    Tokenizes and cleans a string: contractions are broken off from their base words, punctuation is broken out
    into its own token, junk characters are removed, etc. For this corpus, punctuation is already tokenized, so this
    mainly serves to handle contractions (it's) and break up hyphenated words (crime-land => crime - land)
    :param string: the string to tokenize (one sentence, typicall)
    :return: a string with the same content as the input with whitespace where token boundaries should be, so split()
    will tokenize it.
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\<br \/\>", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`\-]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\,", " , ", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r"\!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\-", " - ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return string


class WordEmbeddings:
    """
    Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
    word in the indexer. The 0 vector is returned if an unknown word is queried.
    """
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding_length(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        """
        Returns the embedding for a given word
        :param word: The word to look up
        :return: The UNK vector if the word is not in the Indexer or the vector otherwise
        """
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of("UNK")]


def read_word_embeddings(embeddings_file: str) -> WordEmbeddings:
    """
    Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
    that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
    word embedding files.
    :param embeddings_file: path to the file containing embeddings
    :return: WordEmbeddings object reflecting the words and their embeddings
    """
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    # Make position 0 the UNK token
    word_indexer.add_and_get_index("UNK")
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx+1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            vector = np.array(float_numbers)
            word_indexer.add_and_get_index(word)
            # Append the UNK vector to start. Have to do this weirdly because we need to read the first line of the file
            # to see what the embedding dim is
            if len(vectors) == 0:
                vectors.append(np.zeros(vector.shape[0]))
            vectors.append(vector)
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


#################
# You probably don't need to interact with this code unles you want to relativize other sets of embeddings
# to this data. Relativization = restrict the embeddings to only have words we actually need in order to save memory
# (but this requires looking at the data in advance).

# Relativize the word vectors to the training set
def relativize(file, outfile, indexer, word_counter):
    # f = open(file, encoding='utf8')
    f = open(file, encoding='iso8859')
    o = open(outfile, 'w')
    voc = []
    for line in f:
        word = line[:line.find(' ')]
        if indexer.contains(word):
            print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in indexer.objs_to_ints.keys():
        if word not in voc:
            print("Missing " + word + " with count " + repr(word_counter[word]))
    f.close()
    o.close()


# Relativizes word embeddings to the datasets
if __name__ == '__main__':
    word_indexer = Indexer()
    # The counter is just to see what the counts of missed words are so we can evaluate our tokenization (whether
    # it's mismatched with the word vector vocabulary)
    word_counter = Counter()
    a=read_and_index_sentiment_examples("data/rt/train.txt", word_indexer, add_to_indexer=True, word_counter=word_counter)
    b=read_and_index_sentiment_examples("data/rt/dev.txt", word_indexer, add_to_indexer=True, word_counter=word_counter)
    c=read_and_index_sentiment_examples("data/rt/test-blind.txt", word_indexer, add_to_indexer=True, word_counter=word_counter)
    d=read_and_index_sentiment_examples("data/imdb/train.txt", word_indexer, add_to_indexer=True, word_counter=word_counter)
    e=read_and_index_sentiment_examples("data/imdb/dev.txt", word_indexer, add_to_indexer=True, word_counter=word_counter)
    f=read_and_index_sentiment_examples("data/imdb/test.txt", word_indexer, add_to_indexer=True, word_counter=word_counter)
    # Uncomment these to relativize vectors to the dataset
    # relativize("data/glove.6B/glove.6B.50d.txt", "data/glove.6B.50d-relativized.txt", word_indexer, word_counter)
    # relativize("data/glove.6B/glove.6B.300d.txt", "data/glove.6B.300d-relativized.txt", word_indexer, word_counter)
    # lst = [a, b, c, d, e, f]
    # max_len = max([max([len(x.indexed_words) for x in y]) for y in lst])
    # print(max_len)