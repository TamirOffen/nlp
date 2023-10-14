import io
import os
import unicodedata
import string
import glob

import torch
import random

# alphabet lower case + upper case + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# turn a unicode string s to plain ascii, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS
    )

def load_data():
    # build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)
    
    # read a file and split into lines
    def read_lines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('/Users/tamiroffen/Documents/nlp/RNN/data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categories


"""
To represent a single letter, we will use a one-hot vector of size 
<1 x N_LETTERS>. A one-hot vector is filled with 0s except for a 1 
at the index of the letter, e.g. "b" = <0,1,0,...,0>.

To make a word, we will join them into a tensor of size
<line_length x 1 x N_LETTERS>.
"""


# find index of letter from ALL_LETTERS, e.g. "b" = 1
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)


# turn a letter to a 1 x N_LETTERS tensor
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


# turn a line into an array of ohe vectors.
# shape = (line_length, 1, N_LETTERS)
def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor


def random_training_example(category_lines, all_categories):
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == "__main__":
    # print("ALL_LETTERS:", ALL_LETTERS)
    # print("Number of letters:", len(ALL_LETTERS))

    # print(f'name: "Ślusàrski"')
    # print(f'name in ascii: {unicode_to_ascii("Ślusàrski")}')

    print(letter_to_tensor("b"))
    print(line_to_tensor("abc"))

    # category_lines, all_categories = load_data()
    # print(all_categories)
    # print(category_lines['Italian'][:5])




