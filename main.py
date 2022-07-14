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
DEFAULT_LAST_CONVOLUTION_LAYER = "conv"
DEFAULT_HOP_LENGTH = 256
DEFAULT_WINDOW_SIZE = 1024
SAMPLE_RATE = 8000
FRAME_SIZE = 60


def windows(data, window_size, noise):
    start = 0

    while start < len(data):
        yield start, start + window_size
        start += (window_size // noise)


def extract_features(sub_dirs, file_ext=""):
    window_size = int((DEFAULT_HOP_LENGTH * (FRAME_SIZE - 1)))
    print("Window SIZE {}".format(window_size))
    log_specgrams = []
    labels = []
    noise = 1

    for l, sub_dir in enumerate(sub_dirs):

        for fn in tqdm(glob.glob('drive/MyDrive/D4/' + sub_dir + "/*"), desc="Loading {}".format(sub_dir)):

            sound_clip, _ = librosa.load(fn, sr=SAMPLE_RATE)

            for s, (start, end) in enumerate(windows(sound_clip, window_size, noise)):

                if (len(sound_clip[start:end]) == window_size):

                    signal = sound_clip[start:end]
                    logspec = librosa.stft(signal, n_fft=WIDOWN_SIZE, hop_length=DEFAULT_HOP_LENGTH, center=True)
                    # logspec = librosa.feature.melspectrogram(signal, n_mels=60, sr=sample_rate, n_fft=WIDOWN_SIZE, hop_length=DEFAULT_HOP_LENGTH)
                    logspec = librosa.power_to_db(numpy.abs(logspec), ref=np.max)
                    logspec = logspec / 80 + 1
                    # print(logspec)
                    # exit()
                    # print("CARREGOU")
                    # from PIL import Image
                    # img = Image.fromarray(np.uint8(logspec.reshape((int(WIDOWN_SIZE/2)+1, FRAME_SIZE)) * 255) , 'L')
                    # img.save("image_Mosquito{}.png".format(s))
                    # return
                    if random.randint(0, 10) < 8:
                        log_specgrams.append(logspec)
                        labels.append(int(sub_dir))
            # return

    features = np.asarray(log_specgrams).reshape(len(log_specgrams), int(WIDOWN_SIZE / 2) + 1, FRAME_SIZE, 1)
    # features = np.asarray(log_specgrams).reshape(len(log_specgrams), 60, 40, 1)
    features = np.array(features, dtype=numpy.float32)
    np_labels = np.array(labels, dtype=numpy.float32)
    unique, counts = np.unique(np_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    return np.array(features), np_labels


class GradCAM:

    def __init__(self):
        self.size_image = DEFAULT_SIZE_FEATURE
        self.neural_model = None
        self.output_image = DEFAULT_OUTPUT_FILE_IMAGE
        self.alpha = DEFAULT_ALPHA
        self.image_color_deep = DEFAULT_IMAGE_COLOR_DEEP
        self.last_convolution_layer_name = DEFAULT_LAST_CONVOLUTION_LAYER
        pass

    def get_image_array(self, image_path):
        image_array = keras.preprocessing.image.load_img(image_path, target_size=self.size_image)
        image_array_map = keras.preprocessing.image.img_to_array(image_array)
        image_array_map = numpy.expand_dims(image_array_map, axis=0)
        return image_array_map

    def make_grad_cam_heatmap(self, image_list, index_predict=None):
        last_conv_layer = self.neural_model.get_layer(self.last_convolution_layer_name).output
        grad_model = tf.keras.models.Model([self.neural_model.inputs], [last_conv_layer, self.neural_model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, classifier = grad_model(image_list)
            if index_predict is None:
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
