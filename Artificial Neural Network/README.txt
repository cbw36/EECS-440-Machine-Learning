Data immediately as NP array

Tree:
    Variable
        Dictionary of lists of nodes

    Function
        Construction of tree takes list of integers corresponding to number of units per layer
        Forward propagation on example
        Backward propagation
        Call_FP_BP takes an example
            runs forward propagation on the example
            runs backward propagation
            update weights
Node:
    Variables
        Dictionary of child to weight
        Dictionary of parent to value
        Net input = 0
        Activated input = 0
        dL_dW = 0

    Functions
        Add child
            Includes add parent
        Change weight, takes 2 inputs
        Add to net input
        Reset node
            Resets net input

Major functions
    Construct tree
    Run example on tree
    Cross validation
    Sigmoid activation function
    Other calculation functions?

Architecture
Main function: ann
    Loads data (option1)
    Convert data to np array
    Construct tree (setting # nodes at each layer)
    Call iterate_network function
iterate_network
    will receive a tree, an example set (as np.array),
    until convergence or max iteration:
        select new example
        forward propagate
        back propagate
        update weights
        check convergence

Weekly Commit History
Commit 1:
    We discussed our main data structures and functions which we mostly noted on this README.
    General design/flow has been agreed upon.

    Jonathan Yau:
        Implemented basic functions of Tree and Node.
        Need to implement the calculation functions of Tree and Node.
        Need to implement back propagation and update weights in Tree.
        Figure out what other functions are necessary in Node:
            Ask children for info, call helper function to do calculation

    Connor Wolfe:
        Implemented main ann method with command line inputs
        ann will receive an exset, and call functions to build ann on the whole set or with cv
        wrote function to build ann for cv (cv_ann) and for whole set (full_ann)
        cv_ann will split the data into folds and then build, full_ann directly builds
        Both functions check for convergence criterion and structure accordingly
        Both construct a tree with appropriate number of nodes/layer and define framework to iterate and update examples
        Wrote function to set input layer given a new example

Commit 2:
    Jonathan Yau:
        Implemented back propagation for tree and node
        Added sigmoid function and calculation functions for node
        Tested Node
        Added basic output function and node test example function

    Connor Wolfe:
        Implement method to compute accuracy, recall, ROC, precision, and FP rate
        Fix backprop to compute recursively from the output, not input
        Add weight decay function
        Add test for convergence in stochastic sense (compare weights before and after iterating entire example set and check if their difference is less than epsilon)
        test and debug example forward and backpropogation.  Getting no errors, but strange output to investigate next week
        Also want to normalize input


Commit 3:
    Jonathan Yau:
        Implemented normalization
        Small call_ann changes
        Debugging
    Connor Wolfe: