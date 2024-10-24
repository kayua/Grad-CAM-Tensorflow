import glob
import librosa
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from keras.layers import MaxPooling2D
from keras.models import model_from_json, Model
from tqdm import tqdm
import argparse
import logging
import os

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_SIZE_FEATURE = (299, 299)
DEFAULT_IMAGE_COLOR_DEEP = 256
DEFAULT_ALPHA = 0.4
DEFAULT_OUTPUT_FILE_IMAGE = "Heat_map.jpg"
DEFAULT_LAST_CONVOLUTION_LAYER = "last_layer"
DEFAULT_HOP_LENGTH = 256
DEFAULT_WINDOW_SIZE = 1024
SAMPLE_RATE = 8000
FRAME_SIZE = 60
DEFAULT_PATH_SOUNDS = 'Dataset/'

# Function to create windows over the audio data
def windows(data, window_size):
    """Generator that yields start and end indices for windowing the data."""
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += window_size

# Function to extract spectrogram features from audio files
def extract_features(sub_dirs):
    """
    Extract spectrogram features from audio files in the given subdirectories.
    
    Args:
        sub_dirs (list): List of subdirectories containing the audio files.

    Returns:
        np.array: Array of spectrogram features.
        list: List of corresponding labels.
    """
    window_size = int((DEFAULT_HOP_LENGTH * (FRAME_SIZE - 1)))
    spectrogram_list = []
    labels_list = []

    for sub_dir in sub_dirs:
        subdir_path = os.path.join(DEFAULT_PATH_SOUNDS, sub_dir)
        try:
            for fn in tqdm(glob.glob(subdir_path + "/*"), desc=f"Loading {sub_dir}"):
                try:
                    sound_clip, _ = librosa.load(fn, sr=SAMPLE_RATE)

                    for start, end in windows(sound_clip, window_size):
                        if len(sound_clip[start:end]) == window_size:
                            signal = sound_clip[start:end]
                            spectrogram = librosa.stft(signal, n_fft=DEFAULT_WINDOW_SIZE, hop_length=DEFAULT_HOP_LENGTH, center=True)
                            spectrogram = librosa.power_to_db(np.abs(spectrogram), ref=np.max)
                            spectrogram = spectrogram / 80 + 1
                            spectrogram_list.append(spectrogram)
                            labels_list.append(sub_dir)
                except Exception as e:
                    logging.error(f"Error processing file {fn}: {e}")
        except Exception as e:
            logging.error(f"Error loading files from {sub_dir}: {e}")

    fft_window = int(DEFAULT_WINDOW_SIZE / 2) + 1
    features = np.asarray(spectrogram_list).reshape(len(spectrogram_list), fft_window, FRAME_SIZE, 1)
    return np.array(features, dtype=np.float32), labels_list

# Class to perform GradCAM on a model
class GradCAM:
    def __init__(self, alpha=DEFAULT_ALPHA, output_image=DEFAULT_OUTPUT_FILE_IMAGE, last_convolution_layer_name=DEFAULT_LAST_CONVOLUTION_LAYER):
        self.neural_model = None
        self.output_image = output_image
        self.alpha = alpha
        self.last_convolution_layer_name = last_convolution_layer_name

    def make_grad_cam_heatmap(self, image):
        """
        Generate GradCAM heatmap for a given image input.

        Args:
            image (np.array): Image for which to generate the heatmap.

        Returns:
            np.array: Heatmap array.
        """
        last_conv_layer = self.neural_model.get_layer(self.last_convolution_layer_name).output
        grad_model = tf.keras.models.Model([self.neural_model.inputs], [last_conv_layer, self.neural_model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, classifier = grad_model(image[None, :, :, :])
            predict_index = tf.argmax(classifier[0])
            class_channel = classifier[:, predict_index]

        gradient_propagation = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(gradient_propagation, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        logging.info(f"Generated heatmap with shape: {heatmap.shape}")
        return heatmap.numpy()

    def save_and_display_gradcam(self, img, heatmap, cam_path="cam.jpg", alpha=0.8):
        """
        Save and display the GradCAM heatmap overlaid on the input image.

        Args:
            img (np.array): Input image.
            heatmap (np.array): GradCAM heatmap.
            cam_path (str): Path to save the superimposed image.
            alpha (float): Alpha blending factor.
        """
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * alpha + img * 256
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)
        logging.info(f"Saved GradCAM image to {cam_path}")

    def load_model(self, prefix_model):
        """
        Load the trained model from JSON and H5 files.

        Args:
            prefix_model (str): Model prefix for loading the JSON and weights files.
        """
        try:
            with open(f'{prefix_model}.json', 'r') as json_file:
                loaded_model_json = json_file.read()
            self.neural_model = model_from_json(loaded_model_json)
            self.neural_model.load_weights(f'{prefix_model}.h5')
            self.neural_model = Model(self.neural_model.input, self.neural_model.layers[-2].output)
            self.neural_model.summary()
            logging.info(f"Model {prefix_model} loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model {prefix_model}: {e}")

# Main function to execute the script
def main(args):
    grad_cam = GradCAM()
    grad_cam.load_model(args.model_prefix)

    features, labels = extract_features(args.sub_dirs)
    if features.size == 0:
        logging.error("No features extracted.")
        return

    image_feature = features[1]
    image_heat_map = grad_cam.make_grad_cam_heatmap(features[0])
    image_heat_map = image_heat_map.reshape((16, 1))
    image_heat_map = image_heat_map[2:16]

    for i in range(2, features.shape[0]):
        heatmap = grad_cam.make_grad_cam_heatmap(features[i])
        heatmap = heatmap.reshape((16, 1))
        heatmap = heatmap[2:16]
        image_feature = np.concatenate((image_feature, features[i]), axis=1)
        image_heat_map = np.concatenate((image_heat_map, heatmap), axis=1)

    grad_cam.save_and_display_gradcam(image_feature, image_heat_map)

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform GradCAM on a neural network model.")
    parser.add_argument('--model_prefix', type=str, required=True, help="Prefix for model JSON and H5 files.")
    parser.add_argument('--sub_dirs', nargs='+', required=True, help="Subdirectories containing audio files.")
    
    args = parser.parse_args()
    main(args)
