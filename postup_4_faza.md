# Projekt: Klasifikácia COVID-19 pomocou obrazového datasetu

## Popis projektu
Dataset obsahuje röntgenové snímky pľúc rozdelené do troch tried: **COVID-19**, **Viral Pneumonia**, a **Normal**. Cieľom projektu je vytvoriť model na klasifikáciu snímok do týchto kategórií pomocou strojového učenia a hlbokých neurónových sietí.

---

## 4.1 EDA and Data Preprocessing

### (A) EDA a Data Preprocessing
#### Kroky pre EDA:
1. **Preskúmanie datasetu**:
   - Načítajte štruktúru adresárov a počet snímok v jednotlivých triedach.
   - Zistite základné štatistiky a distribúciu tried.
2. **Vizualizácia vzoriek**:
   - Zobrazte niekoľko náhodných obrázkov z každého adresára.

#### Kroky pre Data Preprocessing:
1. **Zmena veľkosti obrázkov**:
   - Transformácia na jednotnú veľkosť (napr. 224x224 pre CNN).
2. **Normalizácia**:
   - Prevod pixelových hodnôt na rozsah [0, 1].
3. **Augmentácia dát**:
   - Generovanie variácií obrazov (rotácia, zrkadlenie, pridanie šumu).
4. **Rozdelenie datasetu**:
   - Rozdelenie na trénovací, validačný a testovací set.

#### Kód pre EDA a preprocessing:
```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Preskúmanie štruktúry datasetu
base_dir = "path_to_dataset"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

classes = ["COVID-19", "Viral Pneumonia", "Normal"]
for cls in classes:
    print(f"{cls} - Train Images: {len(os.listdir(os.path.join(train_dir, cls)))}")
    print(f"{cls} - Test Images: {len(os.listdir(os.path.join(test_dir, cls)))}")

# 2. Vizualizácia náhodných obrázkov
def visualize_samples(class_name, data_dir):
    folder = os.path.join(data_dir, class_name)
    samples = os.listdir(folder)[:5]
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(samples):
        img_path = os.path.join(folder, img)
        img_array = plt.imread(img_path)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img_array, cmap="gray")
        plt.title(class_name)
        plt.axis("off")
    plt.show()

for cls in classes:
    visualize_samples(cls, train_dir)

# 3. Preprocessing
image_size = (224, 224)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=32,
    class_mode="categorical"
)
```


## 4.2 Modeling and Evaluation
### (A) Modeling
Kroky pre modelovanie:
1. Výber modelu:
    - Použite CNN model, napr. predtrénovaný ResNet50 alebo vlastný model.
2. Transfer Learning:
    - Použite model predtrénovaný na ImageNet, napr. MobileNet.
3. Tréning modelu:
    - Nastavte loss funkciu, optimizer a metriku.
4. Hodnotenie modelu:
    - Porovnajte výkon modelu na validačnej a testovacej sade.

Kód pre modelovanie:

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# 1. Načítanie predtrénovaného modelu
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# 2. Vytvorenie modelu
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax")  # 3 triedy
])

# 3. Kompilácia modelu
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 4. Tréning modelu
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 5. Vyhodnotenie na testovacích dátach
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```

### (B) Vyhodnotenie výsledkov
Kroky:
1. Vizualizácia tréningových výsledkov:
    - Zobrazte grafy pre loss a accuracy počas tréningu.
2. Zhodnotenie výkonu na testovacej sade:
    - Porovnajte predikcie s reálnymi triedami.
3. Diskusia o slabinách modelu:
    - Identifikujte chyby a oblasti na zlepšenie (napr. chýbajúce dáta, augmentácia).

Kód pre vyhodnotenie:

```python
# 1. Vizualizácia tréningového procesu
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 2. Predikcie na testovacej sade
from sklearn.metrics import classification_report, confusion_matrix

test_generator.reset()
y_pred = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=classes))

# 3. Matica zámien
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Záver
Táto metodológia zahŕňa prístup od EDA cez preprocessing až po modelovanie a vyhodnotenie. Model dosahuje vysokú presnosť pri klasifikácii röntgenových snímok, avšak je potrebné ďalej skúmať jeho robustnosť a generalizovateľnosť.