import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from data_manager import DataManager

import matplotlib.pyplot as plt 

from model import Encoder, Decoder

from absl import app
from absl import flags

flags.DEFINE_integer("sample_size", 10, "samples to test")
flags.DEFINE_string("model", "DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
flags.DEFINE_boolean("use_noise", True, "sample noisey images")
FLAGS = flags.FLAGS


def sample(model, n_samples):
    """Passes n random samples through the model and displays X & X_pred"""
    manager = DataManager()
    _, X = manager.get_batch(n_samples, use_noise=FLAGS.use_noise)
    X_pred = model.predict(X)
    X = np.pad(X, ((0,0),(16,16),(16,16),(0,0)),"constant", constant_values=0 )
    x_dim, y_dim = X_pred[0].shape[0], X_pred[0].shape[1]
    X_stitched = np.reshape(X.swapaxes(0,1), (x_dim, y_dim*n_samples)) # n samples = nombres d images qui y a,, accole les images
    X_pred_stitched = np.reshape(X_pred.swapaxes(0,1), (x_dim, y_dim*n_samples))
    stitched_img = np.vstack((X_stitched, X_pred_stitched))
    plt.imshow(stitched_img, cmap='gray')
    plt.show()

def load_model():
    """Set up and return the model."""
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "trained_model", FLAGS.model))
    model = tf.keras.models.load_model(model_path)

    # holds dimensions of latent vector once we find it
    z_dim = None

    # define encoder
    encoder_in  = tf.keras.Input(shape=(32, 32, 1))
    encoder_out = Encoder(encoder_in)
    encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    # load encoder weights and get the dimensions of the latent vector
    for i, layer in enumerate(model.layers):
        encoder.layers[i] = layer
        if layer.name == "encoder_output":
            z_dim = (layer.get_weights()[0].shape[-1])
            break

    # define encoder
    decoder_in  = tf.keras.Input(shape=(z_dim,))
    decoder_out = Decoder(decoder_in)
    decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

    # load decoder weights
    found_decoder_weights = False
    decoder_layer_cnt = 0
    for i, layer in enumerate(model.layers):
        print(layer.name)
        weights = layer.get_weights()
        if len(layer.get_weights()) > 0:
            print(weights[0].shape, weights[1].shape)
        if "decoder_input" == layer.name:
            found_decoder_weights = True
        if found_decoder_weights:
            decoder_layer_cnt += 1
            print("dec:" + decoder.layers[decoder_layer_cnt].name)
            decoder.layers[decoder_layer_cnt].set_weights(weights)

    encoder.summary()
    decoder.summary()

    return encoder, decoder, model
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    encoder, decoder, autoencoder = load_model()
    sample(autoencoder, FLAGS.sample_size)

if __name__ == '__main__':
    app.run(main)
