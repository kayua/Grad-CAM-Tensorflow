import glob

import librosa
import matplotlib.cm as cm
import numpy
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from tensorflow import keras
from tqdm import tqdm

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


def windows(data, window_size):
    start = 0

    while start < len(data):
        yield start, start + window_size
        start += (window_size // 1)


def extract_features(sub_dirs):

    window_size = int((DEFAULT_HOP_LENGTH * (FRAME_SIZE - 1)))
    spectrogram_list = []
    labels_list = []

    for sub_dir in sub_dirs:

        for fn in tqdm(glob.glob(DEFAULT_PATH_SOUNDS + sub_dir + "/*"), desc="Loading {}".format(sub_dir)):

            sound_clip, _ = librosa.load(fn, sr=SAMPLE_RATE)

            for s, (start, end) in enumerate(windows(sound_clip, window_size)):

                if len(sound_clip[start:end]) == window_size:
                    signal = sound_clip[start:end]
                    spectrogram = librosa.stft(signal, n_fft=DEFAULT_WINDOW_SIZE, hop_length=DEFAULT_HOP_LENGTH,
                                               center=True)
                    spectrogram = librosa.power_to_db(numpy.abs(spectrogram), ref=np.max)
                    spectrogram = spectrogram / 80 + 1
                    spectrogram_list.append(spectrogram)
                    labels_list.append(sub_dir)

    fft_window = int(DEFAULT_WINDOW_SIZE / 2) + 1
    features = np.asarray(spectrogram_list).reshape(len(spectrogram_list), fft_window, FRAME_SIZE, 1)
    features = np.array(features, dtype=numpy.float32)

    return np.array(features), labels_list


class GradCAM:

    def __init__(self):
        self.size_image = DEFAULT_SIZE_FEATURE
        self.neural_model = None
        self.output_image = DEFAULT_OUTPUT_FILE_IMAGE
        self.alpha = DEFAULT_ALPHA
        self.image_color_deep = DEFAULT_IMAGE_COLOR_DEEP
        self.last_convolution_layer_name = DEFAULT_LAST_CONVOLUTION_LAYER
        pass


    def make_grad_cam_heatmap(self, image_list):
        last_conv_layer = self.neural_model.get_layer(self.last_convolution_layer_name).output
        grad_model = tf.keras.models.Model([self.neural_model.inputs], [last_conv_layer, self.neural_model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, classifier = grad_model(image_list)
            predict_index = tf.argmax(classifier[0])
            class_channel = classifier[:, predict_index]

        gradient_propagation = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(gradient_propagation, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]

        heat_map = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heat_map = tf.squeeze(heat_map)
        heat_map = tf.maximum(heat_map, 0) / tf.math.reduce_max(heat_map)

        return heat_map.numpy()

    def save_and_display_grad_cam(self, image_input, heat_map):
        img = keras.preprocessing.image.load_img(image_input)
        img = keras.preprocessing.image.img_to_array(img)
        heat_map = np.uint8((self.image_color_deep - 1) * heat_map)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(self.image_color_deep))[:, :3]
        jet_heatmap = jet_colors[heat_map]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * self.alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img.save(self.output_image)

    def load_model(self, prefix_model):
        json_file = open('{}.json'.format(prefix_model), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.neural_model = model_from_json(loaded_model_json)
        self.neural_model.load_weights('{}.h5'.format(prefix_model))


grad_cam = GradCAM()
grad_cam.load_model("models/model_trained_mosquitos")
features, labels = extract_features(["Aedes", "Noise"])

grad_cam.make_grad_cam_heatmap(features)
