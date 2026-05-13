"""
Project: integrated human & artificial intelligence (1d-cnn)
Topic: object-based LULC classification via spectral histograms
Author: Lasya Venigalla
"""

import time
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from keras import layers, models
from rasterio.mask import mask
from shapely.geometry import mapping
from sklearn.metrics import classification_report

# --- 1. configuration & paths ---
data_paths = {
    "raster": r'C:\DLCNN\Data09022022\WVDallasComposite_2.tif',
    "train_shp": r'C:\PhDProjects\Paper2\NewData\WVDallas_SampleSegmentsNot1616Training.shp',
    "test_shp": r'C:\PhDProjects\Paper2\NewData\WVDallas_SampleSegmentsNot1616Testing.shp'
}

bins = 64
classes = ['building', 'grass', 'river', 'road', 'shadow', 'tree', 'water']
class_map = {24:0, 71:1, 11:2, 21:3, 92:4, 41:5, 12:6}

# --- 2. preprocessing utilities ---

def extract_raster_values(geom, src_img, index):
    """extracts raw pixel values from a specific polygon geometry."""
    pgeoms = [mapping(geom[index])]
    out_image, _ = mask(src_img, pgeoms, crop=True)
    
    # filter out 0-value background (masking artifacts)
    img_values = []
    for i in range(out_image.shape[1]):
        for j in range(out_image.shape[2]):
            if not (out_image[:, i, j].tolist().count(0) == src_img.count):
                img_values.append(out_image[:, i, j])
    return np.array(img_values)

def generate_histogram(pixel_values, bands, bins):
    """converts raw spectral data into 1d probability density histograms."""
    freq = []
    for i in range(bands):
        hist, _ = np.histogram(pixel_values[:, i], range=(0, 1), bins=bins, density=True)
        hist = (hist / sum(hist))
        hist[hist == 0] = 0.01  # smoothing zero values
        freq.append(hist)
    return freq

def process_segments(geoms, src_img, bands):
    """pipeline to convert shapefile geometries to histogram arrays."""
    all_freqs = []
    for n in range(geoms.size):
        raw_values = extract_raster_values(geoms, src_img, n)
        hist = generate_histogram(raw_values, bands, bins)
        # zip bands to align features correctly for 1d-cnn input
        all_freqs.append(list(zip(*hist)))
    return np.asarray(all_freqs)

# --- 3. data loading & normalization ---

print("--- loading and normalizing data ---")
with rasterio.open(data_paths["raster"]) as src:
    raster = src.read()
    # min-max scaling (0-1)
    raster_norm = (raster - np.min(raster)) / (np.max(raster) - np.min(raster))
    raster_norm[raster_norm == 0] = 0.0000001
    
    # temporary update to source for processing
    num_bands = src.count

# load shapefiles
train_gdf = gpd.read_file(data_paths["train_shp"])
test_gdf = gpd.read_file(data_paths["test_shp"])

# process histograms
x_train = process_segments(train_gdf.geometry.values, src, num_bands)
x_test = process_segments(test_gdf.geometry.values, src, num_bands)

# prepare labels
y_train = tf.convert_to_tensor([class_map[l] for l in train_gdf['Value']], dtype=tf.int32)
y_test = tf.convert_to_tensor([class_map[l] for l in test_gdf['Value']], dtype=tf.int32)

# --- 4. model architecture (1d-cnn) ---

model = models.Sequential([
    layers.Conv1D(128, 3, padding='same', activation='relu', input_shape=x_train.shape[1:]),
    layers.MaxPooling1D(2),
    layers.Dropout(0.2),
    
    layers.Conv1D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.2),
    
    layers.Conv1D(32, 3, activation='relu'),
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='softplus'),
    layers.Dense(7, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# --- 5. training & evaluation ---

start_time = time.time()
history = model.fit(x_train, y_train, epochs=35, batch_size=32, verbose=1)
train_duration = time.time() - start_time

print("--- training complete ---")

# evaluate and predict
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(x_test), axis=1)

# final reporting
print("test accuracy: ", test_acc)
print(classification_report(y_test, y_pred, target_names=classes))
