## About ##

This repo hosts my implementations of various machine models. Current models include
* Linear Classifier trained with backpropogation as taught in Andrew Ng's Deep Learning course
* L-Layer Feed Forward Neural Network as taught in Andrew Ng's Deep Learning course

Models can be run standalone with default parameters or through the application. Run gui.py

## Testing ##

Each model file contains a main() definition and can be run standalone. 

## Roadmap ##

* Adding application front end
    * Driver  
        * Refactor to be a QT worker
        * Get progress updates from commands
        * Expand model_frame class to hold state information, clean up interface
    * GUI
        * Hide implementations for processing input fields for parameters
        * Create worker threads to dispatch commands
    * Generally define and clean up interfaces
    * Solidify responsibility for state data
    * Add tab for loading a trained model and testing it on a test set
* Gradient Checking
* Gradient descent with momentum, RMSProp, ADAM implementations
* Batch running models with parallelization, result analysis
* Implement a few regularization methods
* Implement tunable learning rate methods
    * Tie learning rate decay to performance checking, try some "smart" tuning while running
* Add more weight initialization options
* Benchmarks against tensorflow implementations for correctness and performance checking
* Unit tests, defensive checks around datatypes
* Batch Normalization implementation
* Expand to multi-class models, softmax
* Dataset Features
    * Shuffle datasets around before each run, split into training, dev, tests sets
    * Batching
    * Synthesize new data examples from existing - rotations, blur, crops
    * Scale, standardize, analyze mean and variance
* Option to save model parameters, load models from file
* Interested Models:
    * RNN
    * CNN
    * NEAT, hyperNEAT, ES-HyperNEAT
    * Graph NN
    * Reservoir computing, NVAR
    * GANs
    * ...