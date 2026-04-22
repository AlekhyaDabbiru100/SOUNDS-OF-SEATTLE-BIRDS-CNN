# Necessary libraries
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical
import librosa, librosa.display
from tensorflow.keras.models import load_model
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical



# loading the file
data = h5py.File('birds/bird_spectrograms.hdf5', 'r')

# Getting the bird species codes
bird_names = list(data.keys())
print(bird_names)

#labeling them for readability
names = {
    'amecro': 'American Crow',
    'amerob': 'American Robin',
    'bewwre': 'Bewicks Wren',
    'bkcchi': 'Black capped Chickadee',
    'daejun': 'Dark eyed Junco',
    'houfin': 'House Finch',
    'houspa': 'House Sparrow',
    'norfli': 'Northern Flicker',
    'rewbla': 'Red winged Blackbird',
    'sonspa': 'Song Sparrow',
    'spotow': 'Spotted Towhee',
    'whcspa': 'White crowned Sparrow'
}
print(names)

# Displaying the names of bird species and their shapes
print("All bird species:")
all_birds = [names[code] for code in bird_names]
for bird in all_birds:
    print(bird)
for code in bird_names:
    shape = data[code].shape      
    print(f"{code:7s} {names.get(code, 'Unknown'):25s} {shape}")
    
# Plotting spectrograms for each specie
n_cols, n_rows = 3, math.ceil(len(bird_names) / 3)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
axes = axes.flatten()
for i, code in enumerate(bird_names):
        spectro = data[code][:, :, 0]        
        axes[i].imshow(spectro, aspect="auto", origin="lower", cmap="gray")
        axes[i].set_title(f"{code}  |  {names.get(code, '')}", fontsize=9)
        axes[i].set_xlabel("Time bins")
        axes[i].set_ylabel("Frequency bins")

for j in range(i + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout()
plt.show()

#EDA
# plot of spectogram length by bird species
length_birds = [ data[code].shape[2] for code in bird_names ]

plt.figure(figsize=(7,7))
plt.barh(all_birds, length_birds, color="yellow")
plt.xlabel("Clip length (timeframes)")
plt.title("Spectrogram length by bird species")
plt.tight_layout()
plt.show()


# Song sparrow vs White crowned sparrow for binary classification, reshaping them.
with data as f:
    song_sparrow = f['sonspa'][...].astype(np.float32) / 255.0
    white_sparrow = f['whcspa'][...].astype(np.float32) / 255.0

    # making into 2-d
    song_sparrow = np.transpose(song_sparrow, (2, 0, 1))
    white_sparrow = np.transpose(white_sparrow, (2, 0, 1))

    # making both the classes the same length
    minimum_length = min(song_sparrow.shape[0], white_sparrow.shape[0])
    song_sparrow = song_sparrow[:minimum_length]
    white_sparrow = white_sparrow[:minimum_length]

    # Labels, 0 for Song Sparrow, 1 for white crowned sparrow
    song_one = np.zeros(minimum_length, dtype=np.uint8)
    white_one = np.ones(minimum_length, dtype=np.uint8)

    # Combining and shuffle 
    x_class = np.concatenate([song_sparrow, white_sparrow], axis=0)
    y_class = np.concatenate([song_one, white_one], axis=0)
    x_class = x_class[..., np.newaxis]
    x_class, y_class = shuffle(x_class, y_class, random_state=42)

    # Splitting 30% train, 40% test, 30% validation set
    x_one, x_test, y_one, y_test = train_test_split(x_class, y_class, test_size=0.4, stratify=y_class, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_one, y_one, test_size=0.5, stratify=y_one, random_state=42)
    
# binary CNN model
model_one = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_one.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model_one.summary()

# Training the binary model.
callbacks_one = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("lecture_cnn_sparrows.keras", save_best_only=True)
]

history = model_one.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=100,
    batch_size=32,
    callbacks=callbacks_one,
    verbose=1
)

# Plotting the binary model training
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.show()

# Binary model evaluation
loss, acc, auc = model_one.evaluate(x_test, y_test, verbose=0)
print(f"\n Final Test Accuracy: {acc:.4f} | AUC: {auc:.4f}")

probability_one = model_one.predict(x_test)
prediction_one = (probability_one > 0.5).astype(int).flatten()

# Binary model's confusion matrix
conf_matrix = confusion_matrix(y_test, prediction_one)
matrix_one = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Song Sparrow', 'White crowned Sparrow'])
matrix_one.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# MULTI-CLASS MODEL
x_all, y_all = [], []   
all_birds  = [] 

with data as f:
    for idx, key in enumerate(f.keys()):           
        spec = f[key][...].astype(np.float32) / 255.0  
        spec = np.transpose(spec, (2, 0, 1))         
        x_all.append(spec)
        y_all.append(np.full(spec.shape[0], idx, dtype=np.uint8))
        all_birds.append(key)

# combining and shuffling
x_class = np.concatenate(x_all, axis=0)
y_class = np.concatenate(y_all, axis=0)
x_class = x_class[..., np.newaxis]

# Splotting the data
x_class, y_class = shuffle(x_class, y_class, random_state=42)

x_one, x_test, y_one, y_test = train_test_split(x_class, y_class, test_size=0.4, stratify=y_class, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_one, y_one, test_size=0.5, stratify=y_one, random_state=42)
    
x_multi, x_test, y_multi, y_test = train_test_split(x_class, y_class,
                                               test_size=0.40, stratify=y_class, random_state=42)

x_train, x_valid, y_train, y_valid = train_test_split(x_multi, y_multi,
                                                     test_size=0.50, stratify=y_multi, random_state=42)

num_class = len(all_birds)
print("Input shape: ", x_train.shape[1:], " | classes:", num_class)

# Multi-class neural network
model_two = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  
    tf.keras.layers.Dense(num_class, activation='softmax')
])
model_two.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_two.summary()

# Training the multi-class model
callbacks_two = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('multiclass_birds.keras', save_best_only=True)
]

history = model_two.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=150,
    batch_size=32,
    callbacks=callbacks_two,
    verbose=1
)

# Plotting the multi-class model training
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Model evaluation of multi-class model
loss, acc = model_two.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc:.4f}")

probability_two  = model_two.predict(x_test)
prediction_two = np.argmax(probability_two , axis=1)
#Confusion matrix of the multi-class model
conf_mat = confusion_matrix(y_test, prediction_two)

ConfusionMatrixDisplay(conf_mat, display_labels=all_birds).plot(
    xticks_rotation=90, cmap=plt.cm.Blues)
plt.title('Multiclass Confusion Matrix')
plt.tight_layout()
plt.show()

# Precisio, recalll, f1 score per class
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    prediction_two,
    zero_division=0
)

df = pd.DataFrame({
    "Class": all_birds,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})
print(df)


# EXTERNAL TEST DATA
audios = {
    "test1": "/Users/alekh/Desktop/birds/test1.wav",
    "test2": "/Users/alekh/Desktop/birds/test2.wav",
    "test3": "/Users/alekh/Desktop/birds/test3.wav"
}

# loading the waveforms
def waveforms(path, name):
    audio, sr = librosa.load(path, sr=22050)  
    # making plots of the waveforms
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"waveform {name}")
    plt.xlabel("time (secs)")
    plt.ylabel("amplitude")
    plt.show()
    
    return audio, sr


audio_list = {}
for name, path in audios.items():
    audio, sr = waveforms(path, name)
    audio_list[name] = (audio, sr)

# plotting the spectrograms from the time slices
def generate_spectrograms(audio, sr, start_sec, end_sec, name):
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    window = audio[start_sample:end_sample]

    S = librosa.feature.melspectrogram(y=window, sr=sr, n_fft=2048, hop_length=512, n_mels=256)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap= 'gray')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram of {name}')
    plt.show()

time_windows = {
    "test1": (15, 17),
    "test2": (0, 2),
    "test3": (2, 4)
}

for name, (audio, sr) in audio_list.items():
    start_sec, end_sec = time_windows[name]
    generate_spectrograms(audio, sr, start_sec, end_sec, name)

# Storing the spectrograms in HDF5 format
def spectro_array(audio, sr, start_sec, end_sec, n_mels=256, n_fft=2048, hop_length=512):
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    window = audio[start_sample:end_sample]
    S = librosa.feature.melspectrogram(y=window, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

with h5py.File('/Users/alekh/Desktop/birds/spectrograms.h5', 'w') as hf:
    for name, (audio, sr) in audio_list.items():
        start_sec, end_sec = time_windows[name]
        S_DB = spectro_array(audio, sr, start_sec, end_sec)
        hf.create_dataset(name, data=S_DB)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='grey')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram of {name}')
        plt.show()

# Test Data pre-processing
spectro_path = '/Users/alekh/Desktop/birds/bird_spectrograms.hdf5'
audios = {
    "test1": "/Users/alekh/Desktop/birds/test1.wav",
    "test2": "/Users/alekh/Desktop/birds/test2.wav",
    "test3": "/Users/alekh/Desktop/birds/test3.wav"
}
time_windows = {
    "test1": (15, 17),
    "test2": (0, 2),
    "test3": (2, 4)
}

spectrogram_list = []
labels = []
with h5py.File(spectro_path, 'r') as f:
    species = list(f.keys())
    for key in species:
        data = f[key][...]
        if data.ndim == 3:
            for i in range(data.shape[2]):
                spectrogram_list.append(data[:, :, i])
                labels.append(key)
        elif data.ndim == 2:
            spectrogram_list.append(data)
            labels.append(key)
        else:
            raise ValueError(f"ndim={data.ndim}")

spectrogram_list = np.array(spectrogram_list)
labels = np.array(labels)
label_encoder = LabelEncoder().fit(labels)
labels_encoded = label_encoder.transform(labels)
labels_onehot = to_categorical(labels_encoded)
mel_bins, time_frames = spectrogram_list.shape[1], spectrogram_list.shape[2]
spectrogram_list = spectrogram_list.reshape(-1, mel_bins, time_frames, 1).astype(np.float32) / 255.0

model_three = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(mel_bins, time_frames, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(species), activation='softmax')
])
model_three.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_three.fit(spectrogram_list, labels_onehot, epochs=5, batch_size=32)

#Predicting the unseen clips
test_data = []
for name, path in audios.items():
    audio, sr = librosa.load(path, sr=22050)
    start, end = time_windows[name]
    clip = audio[int(start*sr):int(end*sr)]
    S = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=2048, hop_length=512, n_mels=mel_bins)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < time_frames:
        S_db = np.pad(S_db, ((0,0),(0, time_frames - S_db.shape[1])), 'constant')
    else:
        S_db = S_db[:, :time_frames]
    test_data.append(S_db)

test_data = np.array(test_data).reshape(-1, mel_bins, time_frames, 1).astype(np.float32) / 255.0
test_predictions = model_three.predict(test_data, verbose=0)
test_one = np.argmax(test_predictions, axis=1)
test_class = label_encoder.inverse_transform(test_one)
for i, pred in enumerate(test_class, 1):
    print(f"Test Spectrogram {i} predicted as {pred}")

# Predicting the top three classes
bird_codes = label_encoder.classes_
bird_names = [names[c] for c in bird_codes]

hop = int(time_frames * 0.5)
rows = []

for clip_name, path in audios.items():
    y, sr = librosa.load(path, sr=22050)
    s, e = time_windows[clip_name]
    seg = y[int(s*sr):int(e*sr)]

    S = librosa.feature.melspectrogram(
        y=seg, sr=sr,
        n_fft=2048, hop_length=512, n_mels=mel_bins
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = np.clip((S_db + 80) / 80, 0, 1)

    n_cols = S_norm.shape[1]
    if n_cols < time_frames:
        pad = time_frames - n_cols
        S_norm = np.pad(S_norm, ((0,0),(0,pad)), mode='constant')
        n_cols = time_frames

    starts = range(0, n_cols - time_frames + 1, hop)
    patches = np.stack([S_norm[:, i:i+time_frames] for i in starts])[..., None]

    P = model_three.predict(patches, verbose=0)
    average = P.mean(axis=0)
    top_three = np.argsort(average)[-3:][::-1]

    rows.append({
        "clip":         clip_name,
        "top1_species": bird_names[top_three[0]],
        "top1_probability":    float(average[top_three[0]]),
        "top2_species": bird_names[top_three[1]],
        "top2_probabaility":    float(average[top_three[1]]),
        "top3_species": bird_names[top_three[2]],
        "top3_probabality":    float(average[top_three[2]]),
    })

df_top3 = pd.DataFrame(rows)
print("The top 3 external clip predictions")
display(df_top3)
