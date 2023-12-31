# Traittement Automatique des Langues
## TP1 

### Récapitulatif du code

* is_number :  Vérifie si une chaîne est un nombre entier.
* filter_word : Filtre un mot en supprimant les liens, mentions et ponctuations.
* convert_to_svm_format : Convertit un message en format SVM en utilisant un dictionnaire de numéros uniques pour les mots.
* process_corpus : Traite un corpus en créant un lexique de mots uniques et en convertissant les messages en format SVM.
* train_model : Entraîne un modèle SVM en utilisant LibLinear en spécifiant des paramètres tels que la valeur de régularisation (c_value) et la tolérance (e_value) fixé par le sujet de TP
* predict_model : Effectue des prédictions en utilisant un modèle SVM préalablement entraîné.
* apply_classifier :
  * Traite les corpus d'entraînement, de validation et de test en utilisant process_corpus().
  * Entraîne un modèle SVM en utilisant train_model() avec le corpus d'entraînement.
  * Effectue des prédictions sur les corpus de validation et de test en utilisant predict_model() avec le modèle SVM précédemment entraîné.
* main : Le script principal main() est appelé si le fichier est exécuté en tant que programme. Il spécifie le chemin vers le dossier LibLinear, traite les fichiers de données d'entraînement, de validation et de test, puis applique le classifieur SVM à ces données.


## Résultat d'exécution

#### Commande : liblinear-2.47/train -c 4 -e 0.1 -v 5 .\SVM\twitter-2013train-A.svm

Résultat :

..............................

optimization finished, #iter = 300

Objective value = -1046.615681

nSV = 4890


WARNING: reaching max number of iterations

Switching to use -s 2


init f 1.055e+03 |g| 1.942e+02

iter  1 f 1.050e+03 |g| 8.783e+01 CG   2 step_size 1.00e+00

iter  2 f 1.049e+03 |g| 5.086e+01 CG   4 step_size 5.00e-01

iter  3 f 1.049e+03 |g| 2.105e+01 CG   2 step_size 1.00e+00 

..............................

optimization finished, #iter = 300

Objective value = -432.048298

nSV = 3412


WARNING: reaching max number of iterations

Switching to use -s 2


init f 4.330e+02 |g| 2.748e+01

..............................

optimization finished, #iter = 300

Objective value = -904.581244

nSV = 4676


WARNING: reaching max number of iterations

Switching to use -s 2


init f 9.362e+02 |g| 4.207e+02

iter  1 f 9.078e+02 |g| 6.769e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -1095.035390

nSV = 4930


WARNING: reaching max number of iterations

Switching to use -s 2


init f 1.103e+03 |g| 1.455e+02

iter  1 f 1.100e+03 |g| 1.088e+02 CG   2 step_size 1.00e+00

iter  2 f 1.100e+03 |g| 2.451e+01 CG   4 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -933.005036

nSV = 4600


WARNING: reaching max number of iterations

Switching to use -s 2


init f 9.388e+02 |g| 1.238e+02

iter  1 f 9.360e+02 |g| 3.114e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -456.031791

nSV = 3389


WARNING: reaching max number of iterations

Switching to use -s 2


init f 4.630e+02 |g| 1.041e+02

iter  1 f 4.585e+02 |g| 4.246e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -1074.746306

nSV = 4981


WARNING: reaching max number of iterations

Switching to use -s 2


init f 1.079e+03 |g| 9.441e+01

iter  1 f 1.077e+03 |g| 4.480e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -942.404656

nSV = 4660


WARNING: reaching max number of iterations

Switching to use -s 2


init f 9.428e+02 |g| 1.957e+01

..............................

optimization finished, #iter = 300

Objective value = -441.515997

nSV = 3453


WARNING: reaching max number of iterations

Switching to use -s 2


init f 4.450e+02 |g| 9.866e+01

iter  1 f 4.436e+02 |g| 3.884e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -1069.364306

nSV = 4929


WARNING: reaching max number of iterations

Switching to use -s 2


init f 1.077e+03 |g| 8.544e+01

iter  1 f 1.075e+03 |g| 4.127e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -900.205170

nSV = 4582


WARNING: reaching max number of iterations

Switching to use -s 2


init f 9.067e+02 |g| 7.544e+01

iter  1 f 9.051e+02 |g| 4.351e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -447.958019

nSV = 3314


WARNING: reaching max number of iterations

Switching to use -s 2


init f 4.490e+02 |g| 3.773e+01

..............................

optimization finished, #iter = 300

Objective value = -1072.477148

nSV = 4967


WARNING: reaching max number of iterations

Switching to use -s 2


init f 1.087e+03 |g| 1.465e+02

iter  1 f 1.080e+03 |g| 1.272e+02 CG   2 step_size 1.00e+00

iter  2 f 1.079e+03 |g| 2.031e+01 CG   4 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -919.459445

nSV = 4619


WARNING: reaching max number of iterations

Switching to use -s 2


init f 9.270e+02 |g| 1.262e+02

iter  1 f 9.244e+02 |g| 3.733e+01 CG   3 step_size 1.00e+00

..............................

optimization finished, #iter = 300

Objective value = -450.114093

nSV = 3462

WARNING: reaching max number of iterations

Switching to use -s 2

init f 4.560e+02 |g| 1.555e+02

iter  1 f 4.527e+02 |g| 9.102e+01 CG   2 step_size 1.00e+00

iter  2 f 4.527e+02 |g| 6.335e+01 CG  15 step_size 3.12e-02

Cross Validation Accuracy = 63.2383%

#### Commande : liblinear-2.47/train -c 4 -e 0.1 .\SVM\twitter-2013train-A.svm .\model\tweets.model  

..............................

optimization finished, #iter = 300

Objective value = -1324.253792

nSV = 5619

WARNING: reaching max number of iterations

Switching to use -s 2

init f 1.338e+03 |g| 1.544e+02

iter  1 f 1.330e+03 |g| 7.009e+01 CG   2 step_size 1.00e+00

..............................

optimization finished, #iter = 300


Objective value = -621.752186


nSV = 4085


WARNING: reaching max number of iterations

Switching to use -s 2


init f 6.246e+02 |g| 4.532e+01

..............................

optimization finished, #iter = 300

Objective value = -1565.661496

nSV = 6019


WARNING: reaching max number of iterations

Switching to use -s 2


init f 1.577e+03 |g| 2.022e+02

iter  1 f 1.573e+03 |g| 7.430e+01 CG   2 step_size 1.00e+00

iter  2 f 1.573e+03 |g| 7.142e+01 CG  16 step_size 7.81e-03 

iter  3 f 1.573e+03 |g| 1.752e+01 CG   4 step_size 1.00e+00 



#### Commande : liblinear-2.47/predict .\SVM\twitter-2013train-A.svm  .\model\tweets.model .\out\out.txt

Accuracy = 99.8348% (9668/9684)

