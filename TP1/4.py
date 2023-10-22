import re
import string
import subprocess

def process_corpus(input_filename, output_filename):
    lexicon = []  # Liste pour stocker le lexique
    unique_numbers = {}  # Dictionnaire pour stocker les numéros uniques des mots

    # Fonction pour vérifier si une chaîne est un nombre
    def is_number(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    # Fonction pour filtrer les mots en supprimant les liens et les mentions
    def filter_word(word):
        word = re.sub(r'https?://\S+', '', word)  # Supprimer les liens
        word = re.sub(r'http://\S+', '', word)  # Supprimer les liens
        word = re.sub(r'www\.\S+', '', word)  # Supprimer les liens
        word = re.sub(r'@[A-Za-z0-9_]+', '', word)  # Supprimer les mentions
        word = word.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
        return word

    # Fonction pour convertir un message en format SVM
    def convert_to_svm_format(message, unique_numbers, label):
        # Mappage des étiquettes aux valeurs souhaitées
        if label == "positive":
            svm_label = "1"
        elif label == "neutral":
            svm_label = "0"
        elif label == "negative":
            svm_label = "-1"
        else:
            svm_label = "0"  # Par défaut, si l'étiquette n'est pas reconnue

        svm_line = [f"{svm_label}"]
        words = message.split()
        for word in words:
            if word in unique_numbers:
                svm_line.append(f"{unique_numbers[word]}:{words.count(word)}")
        return " ".join(svm_line)

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

    # Tri des lignes du fichier SVM en fonction de l'index (première colonne)
    sorted_output_filename = 'sorted_' + output_filename
    sort_svm_file(output_filename, sorted_output_filename)

def sort_svm_file(input_file, output_file):
    data = []  # Liste pour stocker les données

    # Lire le fichier SVM
    with open(input_file, 'r') as infile:
        for line in infile:
            data.append(line)

    # Trier les lignes en fonction de l'index
    sorted_data = sorted(data, key=lambda line: int(line.split(" ")[0]))

    # Écrire les données triées dans le fichier de sortie
    with open(output_file, 'w') as outfile:
        for line in sorted_data:
            outfile.write(line)

def train_model(input_svm, output_model, c_value=4, e_value=0.1, liblinear_path="liblinear-2.47"):
    command = [
        f'{liblinear_path}/train',
        f'-c {c_value}',
        f'-e {e_value}',
        input_svm,
        output_model
    ]

    subprocess.run(" ".join(command), shell=True)

def predict_model(input_svm, model_file, output_file, liblinear_path):
    command = [
        f'{liblinear_path}/predict',
        input_svm,
        model_file,
        output_file
    ]

    subprocess.run(" ".join(command), shell=True)

def apply_classifier(train_file, dev_file, test_file, model_file, liblinear_path):
    train_model(train_file, model_file, liblinear_path)
    predict_model(dev_file, model_file, 'model/dev_out.txt', liblinear_path)
    predict_model(test_file, model_file, 'model/test_out.txt', liblinear_path)

def main():
    # Exemple d'utilisation
    liblinear_path = 'liblinear-2.47'  # Spécifiez le chemin complet vers le dossier de LibLinear
    process_corpus('data_tp1/twitter-2013train-A.txt', 'SVM/twitter-2013train-A.svm')
    process_corpus('data_tp1/twitter-2013dev-A.txt', 'SVM/twitter-2013dev-A.svm')
    process_corpus('data_tp1/twitter-2013test-A.txt', 'SVM/twitter-2013test-A.svm')
    #apply_classifier('SVM/twitter-2013train-A.svm', 'SVM/twitter-2013dev-A.svm', 'SVM/twitter-2013test-A.svm', 'tweets.model', liblinear_path)

if __name__ == "__main__":
    main()
