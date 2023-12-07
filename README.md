## About ##

This repo hosts my implementations of various machine models. Current models include
* Linear Classifier trained with backpropogation as taught in Andrew Ng's Deep Learning course
* L-Layer Feed Forward Neural Network as taught in Andrew Ng's Deep Learning course

## Testing ##

Each model file contains a main() definition and can be run standalone. 
Results are stored in results/logs/ff_nn.log and a cost plot can be found in /results/logs and /results/plots
An example log entry and plot can be found in each.

There is also an application which can be used by running /app/gui.py

## Roadmap ##

* Adding application front end
    * Generally define and clean up interfaces
    * Solidify responsibility for state data
    * Add tab for loading a trained model and testing it on a test set
    * Hide implementations in app class
    * Batch running models with parallelization, result analysis
* Model Functionality
    * Gradient Checking
    * Gradient descent with momentum, RMSProp, ADAM implementations
    * Implement a few regularization methods
    * Implement tunable/decaying learning rate methods
    * Tie learning rate decay to performance checking, try some "smart" tuning while running
    * Batch Norm
    * Expand to multi-class models, softmax
* Benchmarks against tensorflow implementations for correctness and performance checking
* Unit tests, defensive checks around datatypes
* Option to save model parameters, load models from file
* Dataset Features
    * Shuffle datasets around before each run, split into training, dev, tests sets
    * Batching
    * Synthesize new data examples from existing - rotations, blur, crops
    * Scale, standardize, analyze mean and variance
* Interested Models:
    * RNN
    * CNN
    * NEAT, hyperNEAT, ES-HyperNEAT
    * Graph NN
    * Reservoir computing, NVAR
    * GANs
    * ...

## Application UI ##
Main tab with default values
![alt text](https://github.com/DRodriq/ML-Models/blob/main/docs/proj_pic1.png)
Main tab with loaded dataset and model
![alt text](https://github.com/DRodriq/ML-Models/blob/main/docs/proj_pic2.png)
Live cost plot
![alt text](https://github.com/DRodriq/ML-Models/blob/main/docs/proj_pic3.png)