import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# Define paths and constants
DATA_INPUT_PATH = r"C:\Users\blacb\Documents\GitHub\JojoPose\csvFiles"
MODEL_SAVE_PATH = r"C:\Users\blacb\Documents\GitHub\JojoPose\models\model.h5"
METADATA_SAVE_PATH = r"C:\Users\blacb\Documents\GitHub\JojoPose\models\landmark_metadata.pkl"
NUM_POSES = 10

# Load CSV files from the data folder
files = list(Path(DATA_INPUT_PATH).glob("*.csv"))
if not files:
    raise ValueError("No CSV files found in the data folder.")

# Create dataframes
dfs = []
for file in files:
    df = pd.read_csv(str(file))
    dfs.append(df)
df = pd.concat(dfs, axis=0)

# Assume each CSV contains 33 landmarks
# And 1 
landmark_columns = df.columns[df.columns.get_loc("Landmark_0_x") : df.columns.get_loc("Landmark_32_z") + 1]
pose_columns = "Label"

print("Landmarks:", list(landmark_columns))
print("Poses:", list(pose_columns))

# Prepare features (X) and labels (y)
X = df[landmark_columns].values    # Shape: (num_samples, 99)
y = df[pose_columns].astype(int).values  # Shape: (num_samples, {amount of poses})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the landmark features/standardise them
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Reshape for Conv1D (Batch, length, channels)
X_train = X_train.reshape(-1, 99, 1)
X_test = X_test.reshape(-1, 99, 1)



# Build a 1D CNN for jojo poses
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(99, 1)),

    # Block 1
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    # Block 2
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    # Block 3
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalMaxPooling1D(),

    # Dense head
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),

    # Final classification
    tf.keras.layers.Dense(NUM_POSES, activation='softmax')
])

# Use sparse categorical crossentropy to avoid one_hot_encoding
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32
)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save model + metadata (scaler, landmark names)
model.save(MODEL_SAVE_PATH)
with open(METADATA_SAVE_PATH, 'wb') as file:
    pickle.dump({
        'scaler': scaler,
        'landmark_columns': list(landmark_columns),
        'pose_labels': list(range(NUM_POSES))
    }, file)

print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"Metadata saved to {METADATA_SAVE_PATH}")

