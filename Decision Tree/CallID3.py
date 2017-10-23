import numpy as np
from mldata import *
from Node import *
from AttributeHelperFunctions import *
from ID3 import *
from GainCalculations import *
import random

random.seed(12345)  # Output same random number



def run_cv_id3(exset, attributes, option3, option4):
    """"
    Cross Validation call to ID3
    First partition data into stratified "folds" or subsets of examples with proportional number of positive and negative labels to full example set
    Then train decision tree on a training set with N-1 folds in it
    And test data on final fold
    Repeat for all permutations of folds
    Allows for accurate testing of data because testing on training set would yield falsely accurate results 

    :param exset: dataset of type ExampleSet
    :param attributes: dict- key: index of attribute, --> value: attribute of type Feature
    :param option3: if nonnegative, set maximum depth of tree.  If 0, grow full tree
    :param option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 

    :return accuracy = fraction of examples correctly classified in testing iteration
    :return size: Number of nodes in tree
    :return max_depth: length of longest sequence of tests from root to leaf
    :return first_feature: name of first feature used to partition data
    """""
    accuracy = 0.0
    folds = 5
    # Calculating divisions for cross validation
    (sorted_exset, split_locations) = cv_sorting(exset, folds)
    for i in range(folds):
        # Splitting up exset for cross validation
        test_exset = ExampleSet(exset.schema)
        for split_locs in range(split_locations[i], split_locations[i + 1]):
            test_exset.append(sorted_exset[split_locs])
        train_exset = ExampleSet(exset.schema)
        for split_locs in range(0, split_locations[i]):
            train_exset.append(sorted_exset[split_locs])
        for split_locs in range(split_locations[i + 1] + 1, exset.__len__()):
            train_exset.append(sorted_exset[split_locs])
        # Convert both training and test sets to np arrays
        train_exset = np.array(train_exset.to_float())
        test_exset = np.array(test_exset.to_float())
        # Run ID3 on training set
        tree = Node()
        tree = ID3(train_exset, attributes, option3, option4, tree)
        fold_accuracy = test_tree(tree, test_exset)
        max_depth = tree.get_max_depth()
        size = tree.get_size()
        first_feature_number = tree.get_attribute().keys()[0]
        first_feature = attributes[first_feature_number].name

        output(fold_accuracy, size, max_depth, first_feature, i)
        accuracy = accuracy + fold_accuracy

    accuracy = accuracy/folds
    return accuracy, size, max_depth, first_feature


def run_id3(exset_ar, attributes, option3, option4):
    """"
    Call to ID3 using full data set
    Cannot accurately test tree without external data set

    :param exset_ar: dataset of type np.array
    :param attributes: dict- key: index of attribute, --> value: attribute of type Feature
    :param option3: if nonnegative, set maximum depth of tree.  If 0, grow full tree
    :param option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 

    :return accuracy = fraction of examples correctly classified in testing iteration
    :return size: Number of nodes in tree
    :return max_depth: length of longest sequence of tests from root to leaf
    :return first_feature: name of first feature used to partition data
    """""
    tree = Node()
    tree = ID3(exset_ar, attributes, option3, option4, tree)
    accuracy = test_tree(tree, exset_ar)
    first_feature_number = tree.get_attribute().keys()[0]
    first_feature = attributes[first_feature_number].name
    max_depth = tree.get_max_depth()
    size = tree.get_size()
    return accuracy, size, max_depth, first_feature


def cv_sorting(exset,num_folds):
    """
    Returns the sorted exset and the stratified split locations as a num_folds+1 list ([a,b,c...], a to b-1 is a fold)
    :param exset: dataset to partition 
    :param num_folds: number of folds to construct, int
    :return    sorted_exset: exset sorted into folds
                split_locations: indices corresponding to start of each fold in sorted_exset
    """""
    # Divide exset by class label - Could be done better
    example_true = ExampleSet(exset.schema)
    example_false = ExampleSet(exset.schema)
    for example in exset:
        if example.__getitem__(-1):
            example_true.append(example)
        else:
            example_false.append(example)

    # shuffle both ExampleSets
    random.shuffle(example_true)
    random.shuffle(example_false)

    # Calculate how many from each ExampleSet per fold
    sorted_exset = ExampleSet(exset.schema)
    true_split = example_true.__len__()/num_folds
    false_split = example_false.__len__()/num_folds
    split_locations = list()

    # Calculate split locations
    for num in range(0, num_folds):
        split_locations.append(num*(true_split+false_split))  # Evenly sized chunks
    split_locations.append(example_true.__len__()+example_false.__len__())  # The remaining examples

    # Insert data based upon split location
    for num_true in range(0, example_true.__len__()):
        sorted_exset.insert(split_locations[num_true / true_split], example_true.__getitem__(num_true))

    for num_false in range(0, example_false.__len__()):
        # print num_false/false_split
        sorted_exset.insert(split_locations[num_false / false_split], example_false.__getitem__(num_false))

    return sorted_exset, split_locations