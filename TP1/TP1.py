import re
import os
import string
import subprocess

def is_number(s):
    """Vérifie si une chaîne est un nombre entier."""
    try:
        int(s)
        return True
    except ValueError:
        return False

def filter_word(word):
    """Filtre un mot en supprimant les liens, mentions et ponctuations."""
    word = re.sub(r'https?://\S+', '', word)  # Supprimer les liens
    word = re.sub(r'http://\S+', '', word)  # Supprimer les liens
    word = re.sub(r'www\.\S+', '', word)  # Supprimer les liens
    word = re.sub(r'@[A-Za-z0-9_]+', '', word)  # Supprimer les mentions
    word = word.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    return word

def convert_to_svm_format(message, unique_numbers, label):
    """Convertit un message en format SVM."""
    if label == "positive":
        svm_label = "1"
    elif label == "neutral":
        svm_label = "0"
    elif label == "negative":
        svm_label = "-1"
    else:
        svm_label = "0"

    svm_line = [f"{svm_label}"]
    words = message.split()
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    word_indices = set([unique_numbers[word] for word in words if word in unique_numbers])
    sorted_indices = sorted(word_indices)
    for word_index in sorted_indices:
        count = word_counts.get(word, 0)
        svm_line.append(f"{word_index}:{count}")
    return " ".join(svm_line)

def process_corpus(input_filename, output_filename):
    """Traite un corpus en créant un lexique et en convertissant les messages en format SVM."""
    lexicon = []  # Liste pour stocker le lexique
    unique_numbers = {}  # Dictionnaire pour stocker les numéros uniques des mots
    corpus = []  # Liste pour stocker les messages du corpus

    with open(input_filename, 'r') as file:
        for line in file:
            words = re.split(r'\s+', line)
            message = ""
            label = words[1]  # Obtenez le label à partir de la deuxième colonne
            for word in words[2:]:
                word = word.lower()
                word = filter_word(word)
                if not is_number(word) and word not in ["positive", "neutral", "negative"]:
                    message += word + " "
                    if word not in lexicon:
                        lexicon.append(word)

            corpus.append((label, message.strip()))

    for i, word in enumerate(lexicon):
        unique_numbers[word] = i + 1

    with open(output_filename, 'w') as outfile:
        for label, message in corpus:
            svm_line = convert_to_svm_format(message, unique_numbers, label)
            outfile.write(svm_line + "\n")

def train_model(input_svm, output_model, c_value=4, e_value=0.1, liblinear_folder="liblinear-2.47"):
    """Entraîne un modèle SVM en utilisant LibLinear."""

    liblinear_path = os.path.join(os.getcwd(), liblinear_folder)
    train_path = os.path.join(liblinear_path, "train")

    # Normalisez le chemin pour résoudre les problèmes de caractères spéciaux
    train_path = os.path.normpath(train_path)

    command = [
        f'"{train_path}"' ,
        f'-c {c_value}',
        f'-e {e_value}',
        input_svm,
        output_model
    ]

    subprocess.run(" ".join(command), shell=True)

def predict_model(input_svm, model_file, output_file, liblinear_folder="liblinear-2.47"):
    """Effectue des prédictions en utilisant un modèle SVM."""

    liblinear_path = os.path.join(os.getcwd(), liblinear_folder)
    predict_path = os.path.join(liblinear_path, "predict")

    # Normalisez le chemin pour résoudre les problèmes de caractères spéciaux
    predict_path = os.path.normpath(predict_path)

    command = [
        f'"{predict_path}"',
        input_svm,
        model_file,
        output_file
    ]

    subprocess.run(" ".join(command), shell=True)

def apply_classifier(train_file, dev_file, test_file, model_file, liblinear_path):
    """Applique le classifieur SVM aux données d'entraînement, de validation et de test."""
    train_model(train_file, model_file, liblinear_path)
    predict_model(dev_file, model_file, 'model/tweets.model', liblinear_path)
    predict_model(test_file, model_file, 'model/tweets.model', liblinear_path)

def main():
    # Exemple d'utilisation
    liblinear_path = 'liblinear-2.47'  # Spécifiez le chemin complet vers le dossier de LibLinear
    process_corpus('data_tp1/twitter-2013train-A.txt', 'SVM/twitter-2013train-A.svm')
    process_corpus('data_tp1/twitter-2013dev-A.txt', 'SVM/twitter-2013dev-A.svm')
    process_corpus('data_tp1/twitter-2013test-A.txt', 'SVM/twitter-2013test-A.svm')
    apply_classifier('SVM/twitter-2013train-A.svm', 'SVM/twitter-2013dev-A.svm', 'SVM/twitter-2013test-A.svm', 'tweets.model', liblinear_path)

if __name__ == "__main__":
    main()
