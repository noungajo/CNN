# CNN
implementation of CNN using tensorflow and keras

## Tensorflow
TensorFlow est une bibliothèque open-source d'apprentissage automatique développée par Google. Elle est principalement utilisée pour la création, l'entraînement et le déploiement de modèles d'apprentissage automatique, en particulier de réseaux de neurones. TensorFlow est l'une des bibliothèques les plus populaires pour le développement d'applications d'intelligence artificielle, de traitement du langage naturel, de vision par ordinateur, de reconnaissance vocale et bien d'autres domaines.

1. **Tenseurs** : Le nom "TensorFlow" provient de la notion de "tenseurs", qui sont des tableaux multidimensionnels utilisés pour stocker et manipuler les données. Les opérations sur les tenseurs sont au cœur de TensorFlow, d'où le nom.

2. **Graphes de calcul** : TensorFlow représente les opérations mathématiques comme des nœuds dans un graphe de calcul. Chaque nœud effectue une opération sur des tenseurs et transmet le résultat à d'autres nœuds. Cela permet de créer et de gérer des modèles de manière modulaire.

3. **Flexibilité** : TensorFlow offre une grande flexibilité pour la création de modèles. Vous pouvez concevoir des réseaux de neurones personnalisés en définissant vos propres architectures, ou utiliser des API de haut niveau comme Keras pour simplifier le processus.

4. **Accélération matérielle** : TensorFlow est compatible avec diverses unités de traitement graphique (GPU) et unités de traitement tensoriel (TPU) pour accélérer les calculs, ce qui en fait un choix idéal pour les tâches gourmandes en calcul.

5. **Communauté active** : TensorFlow bénéficie d'une communauté active de chercheurs, de développeurs et d'entreprises qui contribuent au développement continu de la bibliothèque et partagent leurs connaissances.

6. **Déploiement** : TensorFlow propose des outils pour déployer des modèles dans des applications et des environnements de production, ce qui en fait une solution complète pour l'apprentissage automatique, de la conception à la mise en service.

7. **Compatibilité** : TensorFlow est compatible avec plusieurs langages de programmation, dont Python, C++, Java et d'autres, ce qui le rend accessible à un large public.

### Exemple d'utilisation de tensorflow
Voici un exemple simple d'utilisation de TensorFlow pour créer un modèle de réseau de neurones qui effectue une classification d'images à l'aide de l'ensemble de données MNIST. MNIST est un ensemble de données de chiffres manuscrits largement utilisé pour la classification. Nous allons utiliser TensorFlow 2.x avec l'API Keras, qui est intégrée à TensorFlow.

Tout d'abord, il faut s'assurer d'installer TensorFlow en exécutant `pip install tensorflow`.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger l'ensemble de données MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normaliser les images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Créer un modèle de réseau de neurones
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Aplatir les images 28x28 en un vecteur de 784
    layers.Dense(128, activation='relu'),  # Couche cachée avec 128 neurones et fonction d'activation ReLU
    layers.Dropout(0.2),  # Dropout pour la régularisation
    layers.Dense(10)  # Couche de sortie avec 10 neurones pour les 10 classes (chiffres de 0 à 9)
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=5)

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Dans cet exemple, nous avons chargé l'ensemble de données MNIST, normalisé les images, créé un modèle de réseau de neurones simple avec des couches denses (entièrement connectées), compilé le modèle avec une fonction de perte et une métrique, puis entraîné le modèle sur les données d'entraînement. Enfin, nous avons évalué la précision du modèle sur l'ensemble de test.



## Keras
Keras est une bibliothèque open-source d'apprentissage automatique haut niveau, spécialement conçue pour la création de réseaux de neurones artificiels. Elle est conçue pour être conviviale, modulaire et flexible, ce qui la rend accessible aux développeurs de tous niveaux, des débutants aux experts en apprentissage automatique.

1. **Simplicité d'utilisation** : Keras est réputée pour sa simplicité d'utilisation. Elle offre une interface conviviale pour définir, entraîner et évaluer des modèles d'apprentissage automatique, ce qui permet aux développeurs de se concentrer sur la conception de leurs modèles plutôt que sur les détails techniques.

2. **Modularité** : Keras repose sur le principe de modularité. Vous pouvez facilement combiner des couches pour créer des architectures de réseaux de neurones personnalisées. Elle permet également de connecter des modèles pré-entraînés pour le transfert d'apprentissage.

3. **Compatibilité multi-backend** : Keras est conçu pour être compatible avec plusieurs "backends" d'apprentissage automatique, notamment TensorFlow, Theano et Microsoft Cognitive Toolkit (CNTK). Cela signifie que vous pouvez utiliser Keras avec le backend de votre choix.

4. **Support multiplateforme** : Keras fonctionne sur diverses plates-formes, notamment Windows, macOS et Linux. Elle est également compatible avec plusieurs langages de programmation, notamment Python et R.

5. **Large communauté et documentation** : Keras bénéficie d'une grande communauté d'utilisateurs et de développeurs. Il existe de nombreuses ressources, tutoriels et exemples disponibles en ligne pour aider les utilisateurs à apprendre et à résoudre des problèmes.

6. **Interopérabilité avec TensorFlow** : Depuis la version 2.0, Keras est intégrée à TensorFlow en tant que couche d'API de haut niveau. Cela signifie que Keras est devenue la bibliothèque par défaut pour les modèles de haut niveau dans TensorFlow.

7. **Applications variées** : Keras est adaptée à une large gamme d'applications d'apprentissage automatique, telles que la classification d'images, la détection d'objets, le traitement du langage naturel, la génération de texte, la vision par ordinateur, etc.

8. **Déploiement facilité** : Keras offre des outils pour déployer des modèles dans des applications du monde réel, ce qui en fait une solution complète pour l'apprentissage automatique, de la conception à la mise en service.
### Exemple d'utilisation de keras
Voici un exemple simple d'utilisation de Keras pour créer un modèle de réseau de neurones qui effectue une classification d'images à l'aide de l'ensemble de données MNIST, comme dans l'exemple précédent :

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger l'ensemble de données MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normaliser les images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Créer un modèle de réseau de neurones
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Aplatir les images 28x28 en un vecteur de 784
    layers.Dense(128, activation='relu'),  # Couche cachée avec 128 neurones et fonction d'activation ReLU
    layers.Dropout(0.2),  # Dropout pour la régularisation
    layers.Dense(10, activation='softmax')  # Couche de sortie avec 10 neurones pour les 10 classes (chiffres de 0 à 9) et fonction d'activation softmax
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Fonction de perte pour la classification
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=5)

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Cet exemple est similaire à l'exemple précédent utilisant TensorFlow, mais il utilise spécifiquement l'API Keras de TensorFlow. Nous chargeons l'ensemble de données MNIST, normalisons les images, créons un modèle de réseau de neurones, le compilons et l'entraînons sur les données d'entraînement. Enfin, nous évaluons la précision du modèle sur l'ensemble de test.

## Relation entre Keras et Tensoflow

1. **Keras est une interface de haut niveau** : Keras est une bibliothèque d'apprentissage automatique de haut niveau qui offre une interface conviviale pour la conception, la formation et l'évaluation de modèles de réseaux de neurones. Elle a été conçue pour être simple et intuitive, ce qui la rend idéale pour les débutants en apprentissage automatique. Keras est agnostique par rapport au backend d'apprentissage automatique et peut être utilisée avec différents backends, dont TensorFlow, Theano et Microsoft Cognitive Toolkit (CNTK).

2. **TensorFlow est un backend** : TensorFlow est une bibliothèque d'apprentissage automatique open-source développée par Google. C'est un framework plus bas niveau qui fournit des outils pour la création de modèles d'apprentissage automatique, y compris des réseaux de neurones. À partir de TensorFlow 2.0, Keras a été intégrée en tant qu'API de haut niveau, ce qui signifie que vous pouvez utiliser l'interface Keras pour construire des modèles de réseaux de neurones tout en profitant de la puissance de calcul de TensorFlow en coulisses. En d'autres termes, Keras est maintenant une partie intégrante de TensorFlow, mais peut également être utilisée de manière autonome avec d'autres backends.


