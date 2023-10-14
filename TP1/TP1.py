import re
import string
from liblinear.liblinearutil import *

# Dictionaries for lexicon and word occurrences
lexicon = {}  # Dictionary for the lexicon
occurrences = {}  # Dictionary for word occurrences

# Define the version and input/output files
version = 'train'
input_file_txt = 'data_tp1/twitter-2013' + version + '-A.txt'
output_file_svm = 'data_tp1/twitter-2013' + version + '-A.svm'
output_file_svm_sorted = 'data_tp1/twitter-2013' + version + '-A_sorted.svm'


# Function to check if a string is a number
def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Function to filter words by removing links and mentions
def filter_word(word):
    word = re.sub(r'https?://\S+', '', word)  # Remove links
    word = re.sub(r'http://\S+', '', word)  # Remove links
    word = re.sub(r'www\.\S+', '', word)  # Remove links
    word = re.sub(r'@[A-Za-z0-9_]+', '', word)  # Remove mentions
    word = word.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return word


# Function to load a file and build the lexicon
def load_file(file):
    delimiter = r'\s+'
    word_number = 1

    with file as input_file:
        for line in input_file:
            words = re.split(delimiter, line)
            for word in words:
                word = word.lower()
                word = filter_word(word)
                if not is_number(word) and word not in {"positive", "neutral", "negative"}:
                    if word not in lexicon:
                        lexicon[word] = word_number
                        word_number += 1


# Function to calculate the number of occurrences for each word
def calculate_word_occurrences(file):
    # Initialize the occurrences dictionary to zero for each word in the lexicon
    for word in lexicon:
        occurrences[word] = 0

    # Iterate through each word in the corpus
    with file as input_file:
        for line in input_file:
            words = re.split(r'\s+', line)
            for word in words:
                word = filter_word(word)
                if word in occurrences:
                    occurrences[word] += 1


# Function to convert the source file to SVM format
def convert_to_svm_format(infile):
    class_labels = {"positive": 1, "negative": -1, "neutral": 0}

    output_file = open(output_file_svm, "w")

    for line in infile:
        words = line.strip().split()

        label = words[1]
        if label not in class_labels:
            continue

        # Build the line in SVM format
        svm_line = [str(class_labels[label])]

        # Create a dictionary to store word occurrences
        word_occurrence_dict = calculate_word_occurrences(words)

        # Add word occurrences to the SVM line
        for word, count in word_occurrence_dict.items():
            svm_line.append(f"{lexicon[word]}:{count}")

        # Write the line in SVM format to the output file
        output_file.write(" ".join(svm_line) + "\n")

    sort_lines_by_index(output_file_svm, output_file_svm_sorted)


# Function to calculate word occurrences in a list of words
def calculate_word_occurrences(words):
    word_occurrence_dict = {}
    for word in words[1:]:
        word = filter_word(word.lower())
        if word in lexicon:
            word_occurrence_dict[word] = word_occurrence_dict.get(word, 0) + 1
    return word_occurrence_dict


# Function to sort lines by index
def sort_lines_by_index(input_file, output_file):
    sorted_lines = []

    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Process each line
    for line in lines:
        elements = line.split()
        label = elements[0]
        index_values = elements[1:]
        index_values.sort(key=lambda x: int(x.split(':')[0]))
        sorted_line = label + ' ' + ' '.join(index_values)
        sorted_lines.append(sorted_line)

    # Write the sorted lines to the output file
    with open(output_file, 'w') as f:
        f.writelines('\n'.join(sorted_lines))


# Main function
def main():
    input_file = open(input_file_txt, 'r')
    load_file(input_file)

    # Reopen the file for occurrence calculation
    input_file = open(input_file_txt, 'r')
    convert_to_svm_format(input_file)


if __name__ == "__main__":
    main()
