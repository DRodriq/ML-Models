import tensorflow as tf
import sys
import os
sys.path.insert(1, os.getcwd())
from models import model_stage


class FF_NeuralNet_TensorFlow():
    
    def __init__(self, layer_dims, **kwargs):
        hidden_activation_fn, output_activation_fn,  = "tanh", "sigmoid"
        optimizer, loss = "SGD", "sparse_categorical_crossentropy"
        if('hidden_fn' in kwargs):
            hidden_activation_fn = kwargs["hidden_fn"]
        if('output_fn' in kwargs):
            output_activation_fn = kwargs["output_fn"]
        if('optimizer' in kwargs):
            optimizer = kwargs["optimizer"]
        if('loss' in kwargs):
            loss = kwargs["loss"]

        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(layer_dims[1], activation=hidden_activation_fn, input_shape=(layer_dims[0],)),
        tf.keras.layers.Dense(layer_dims[2], activation=hidden_activation_fn),
        tf.keras.layers.Dense(layer_dims[3], activation=output_activation_fn)
        ])

        self.model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])
        
    def train(self, X, Y, num_iterations=3000, sig=None, log=False):
        self.model.fit(X, Y, epochs=10)

    def test(self, X, Y):
        self.model.evaluate(X, Y)


if __name__ == '__main__':
    stage = model_stage.ModelStage(console_log=True)

    did_load, log = stage.set_dataset("cats", 1)
    assert(did_load==True)
    print(log)

    input_fn = stage.dataset.get("Flattened Training Set").shape[0]

    ffnn = FF_NeuralNet_TensorFlow([input_fn, 5, 5,3])
    
    ffnn.train(stage.dataset.get("Flattened Training Set"), stage.dataset.get("Training Set Labels"))

    ffnn.test(stage.dataset.get("Test Set Data"), stage.dataset.get("Test Set Labels"))

