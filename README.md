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

## CNN
Un CNN, ou Convolutional Neural Network en anglais, est un type de réseau de neurones artificiels spécialement conçu pour le traitement et l'analyse d'images et de données spatiales. Les CNN sont largement utilisés dans des tâches telles que la reconnaissance d'images, la classification d'images, la détection d'objets, la segmentation d'images et bien d'autres domaines de vision par ordinateur. 

1. **Neurones convolutifs** : Un CNN est composé de neurones convolutifs, également appelés filtres ou noyaux, qui sont responsables de la détection des motifs et des caractéristiques dans les images. Ces neurones sont appliqués de manière répétée à des zones locales de l'image pour extraire des informations importantes.

2. **Convolution** : La convolution est l'opération clé d'un CNN. Elle consiste à appliquer les neurones convolutifs à une petite fenêtre (par exemple, 3x3 pixels) de l'image, puis à déplacer cette fenêtre sur toute l'image. Cela permet de créer une carte de caractéristiques qui met en évidence les motifs détectés dans l'image.

3. **Strates de convolution** : Un CNN est généralement composé de plusieurs strates de convolution, chacune comportant plusieurs filtres. Les strates plus profondes sont capables de capturer des caractéristiques de plus haut niveau, tandis que les strates plus superficielles détectent des caractéristiques de bas niveau telles que les bords et les textures.

4. **Strates de sous-échantillonnage (pooling)** : Après la convolution, des strates de sous-échantillonnage sont souvent utilisées pour réduire la dimension de la carte de caractéristiques résultante. Cela aide à réduire la quantité de calcul nécessaire et à rendre le modèle plus efficace.

5. **Couche entièrement connectée** : Après plusieurs strates de convolution et de sous-échantillonnage, un CNN peut comporter une ou plusieurs couches entièrement connectées, similaires à celles d'un réseau de neurones traditionnel. Ces couches sont responsables de la classification finale des caractéristiques extraites.

6. **Fonction d'activation** : Les neurones dans un CNN utilisent généralement une fonction d'activation, comme la fonction ReLU (Rectified Linear Unit), qui introduit une non-linéarité dans le modèle.

7. **Apprentissage profond** : Les CNN sont capables d'apprendre des caractéristiques hiérarchiques à partir des données, ce qui signifie qu'ils peuvent détecter des motifs complexes à partir de caractéristiques plus simples.

8. **Rétropropagation** : L'apprentissage d'un CNN se fait par rétropropagation (backpropagation), où les poids des neurones sont ajustés pour minimiser la fonction de perte par rapport aux étiquettes réelles des données d'entraînement.

### Exemple d'utilisation d'un CNN
Voici un exemple d'utilisation d'un CNN (Convolutional Neural Network) pour la classification d'images en utilisant TensorFlow et Keras. Nous allons créer un CNN simple pour classer des images de chiffres manuscrits à partir de l'ensemble de données MNIST. 

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger l'ensemble de données MNIST
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normaliser les images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Créer un modèle CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Couche de convolution
    layers.MaxPooling2D((2, 2)),  # Couche de sous-échantillonnage
    layers.Conv2D(64, (3, 3), activation='relu'),  # Deuxième couche de convolution
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),  # Aplatir les données
    layers.Dense(64, activation='relu'),  # Couche dense
    layers.Dense(10, activation='softmax')  # Couche de sortie pour 10 classes
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reshape des données pour s'adapter au modèle (ajouter une dimension de canal)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# Entraîner le modèle
model.fit(train_images, train_labels, epochs=5)

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')
```

Cet exemple crée un CNN qui prend des images 28x28 pixels en niveaux de gris en entrée et les classe dans l'une des 10 catégories correspondant aux chiffres de 0 à 9. Le modèle est composé de couches de convolution, de sous-échantillonnage, de couches denses et de couche de sortie softmax. Après l'entraînement, le modèle est capable de classer correctement les chiffres manuscrits. Vous pouvez explorer des architectures plus complexes et des ensembles de données plus vastes pour des tâches de classification d'images plus avancées.

### Relation entre un CNN, Tensorflow et Keras
Les CNN (Convolutional Neural Networks), TensorFlow et Keras sont tous interconnectés de la manière suivante :

1. **TensorFlow** :
   - TensorFlow est un framework d'apprentissage automatique open-source développé par Google.
   - Il fournit des outils pour la création, l'entraînement et le déploiement de modèles d'apprentissage automatique, y compris des réseaux de neurones, tels que les CNN.
   - TensorFlow offre un support natif pour l'inférence matricielle et la gestion des calculs sur les unités de traitement graphique (GPU) pour accélérer les opérations d'apprentissage profond.

2. **Keras** :
   - Keras est une bibliothèque d'apprentissage automatique de haut niveau qui offre une interface conviviale pour la création de modèles de réseaux de neurones.
   - Keras est agnostique par rapport au backend d'apprentissage automatique, ce qui signifie qu'elle peut être utilisée avec différents backends, dont TensorFlow.
   - Depuis TensorFlow 2.0, Keras a été intégrée en tant qu'API de haut niveau dans TensorFlow, devenant ainsi l'API de choix pour la création de modèles de haut niveau dans TensorFlow.

La relation entre les trois est la suivante :

- TensorFlow peut être utilisé de manière autonome pour créer des modèles de réseaux de neurones, y compris des CNN, en utilisant ses propres API et outils.
- Keras peut également être utilisé de manière autonome pour créer des modèles de réseaux de neurones, y compris des CNN, et ce, avec différents backends, dont TensorFlow.
- TensorFlow a intégré l'API Keras en tant que couche de haut niveau, ce qui signifie que les utilisateurs de TensorFlow peuvent maintenant utiliser l'interface Keras pour créer des modèles de réseaux de neurones.

Ainsi, lorsque vous utilisez TensorFlow avec son intégration Keras, vous bénéficiez de la simplicité et de l'expressivité de Keras tout en profitant de la puissance de calcul et de l'écosystème de TensorFlow. Cela permet aux utilisateurs de choisir l'approche qui leur convient le mieux en fonction de leurs besoins et de leur expérience. Les CNN sont couramment utilisés pour le traitement d'images et sont compatibles avec les deux frameworks, ce qui facilite la création de modèles de vision par ordinateur puissants.

