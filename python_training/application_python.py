#read csv

import pandas as pd

#data = pd.read_csv('eg.csv')
#print(data.head())


#Count words, char, lines in text file.

import os

def count(filename):
    with open(filename, 'r') as file:
        text = file.read()

        lines = text.splitlines()
        line_count = len(lines)

        words = text.split()
        word_count = len(words)

        char_count = len(text)

    return line_count, word_count, char_count

l, w, c = count('eg.txt')
print(f'Lines: {l} Words: {w} Char: {c}')



    

