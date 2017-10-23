import numpy as np
from mldata import *
from Node import *
import random
from GainCalculations import *

random.seed(12345)  # Output same random number


def find_best_attribute(examples, attributes, option4):
    """
    Find best attribute to split an example set
    Iterate throug all possible splits and use the one that maximizes information gain
    :param examples: set of examples to split, type np.array
    :param attributes: dict of features: {feature id:Feature}
    :param option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 
    :return: best_attribute: dict of best feature to split on: {feature id:Feature}
    """""
    best_attribute = dict()
    best_entropy = 100
    best_split = None
    for key in attributes:
        cur_att = attributes[key]
        if cur_att.type == 'NOMINAL':
            freq = nominal_attribute(key, examples)
            if option4 == 0:
                cur_entropy = IG(freq)
            else:
                cur_entropy = IGR(freq)
            if cur_entropy < best_entropy:
                best_entropy = cur_entropy
                best_attribute = dict()
                best_split = None
                best_attribute[key] = cur_att

        elif cur_att.type == 'CONTINUOUS':
            val_label_freq = compute_continuous_frequencies(examples, key)  # [[feat_val, [num_pos, num_neg]], [feat_val,[num_pos, num_neg]]...]
            thresholds = compute_thresholds(val_label_freq)  # [thresh1, thresh2, ...]
            cur_entropy, split = find_continuous_entropy(val_label_freq, thresholds, option4)
            if cur_entropy < best_entropy:
                best_entropy = cur_entropy
                best_split = split
                best_attribute = dict()
                best_attribute[key] = cur_att
    return best_attribute, best_split


def create_new_attribute_list(attributes, cur_ind):
    """
    Compute an attribute dict with the attribute just split on removed
    :param attributes: dict of features: {feature id:Feature}
    :param cur_ind: Index of feature to remove from dict
    :return: new_attributes: attributes with current attribute removed
    """""
    new_attributes = dict()
    cur_att_keys = attributes.keys()
    cur_att_keys.remove(cur_ind)
    for key in cur_att_keys:
        new_attributes[key] = attributes[key]
    return new_attributes


def compute_threshold_bins(val_label_freq, thresholds):
    """
    Construct bins of examples that fall between threshold values
    Optimizes algorithm by not having to recurse through entire attribute list for checking each thresholds information gain
    Instead just iterate through bins
    :param val_label_freq: Number of occurences of each value of a given attribute and label pair
    :param thresholds: List of threshold values for a given attribute
    :return:    bins: list of lists of examples that fall within two threshold values
                total_pos: number of positive examples (calculated here to avoid iterating through again)
                total_neg: number of negative examples
    """""
    bins=[]
    i=0
    cur_bin = []
    total_pos = 0
    total_neg = 0

    for val in val_label_freq:
        if i == len(thresholds):
            cur_bin.append(val)
            total_pos += val[1][0]
            total_neg += val[1][1]
        elif val[0] <= thresholds[i]:
            cur_bin.append(val)
            total_pos += val[1][0]
            total_neg += val[1][1]
        else:
            bins.append(cur_bin)
            cur_bin = []
            cur_bin.append(val)
            i=i+1
            total_pos += val[1][0]
            total_neg += val[1][1]
    return bins, total_pos, total_neg


def compute_continuous_frequencies(examples, att_ind):
    """
    Compute Number of occurences of each value of a given attribute and label pair
    :param examples: set of examples to split, type np.array
    :param att_ind: index of attribute were computing on
    :return: value_label_freq: number of occurences of each value of a given attribute and label pair
    """""
    value_label_mat = examples[:, [att_ind, -1]]  # #examples x 2 array, col 1 = feat vals, col 2 = class label
    unique_feat_vals = np.sort(np.unique(value_label_mat[:, 0]))
    value_label_freq = []
    for val in unique_feat_vals:
        ind = np.where(value_label_mat[:, 0] == val)[0]
        classes = value_label_mat[ind, -1]
        num_pos = sum(classes)
        num_neg = len(classes) - num_pos
        value_label_freq.append([val, [num_pos, num_neg]])
    return value_label_freq


def compute_thresholds(value_label_freq):
    """
    Find all threshold values for a given feature.  
    A threshold value is one when the examples are sorted by the feature, two adjascent examples have different class labels
    :param value_label_freq: number of occurences of each value of a given attribute and label pair
    :return: thresholds: list of threshold values for a given feature
    """""
    #find splits
    #value_label_freq = [[feat_val, [num_pos, num_neg]], [feat_val,[num_pos, num_neg]]...]
    thresholds = []
    for i in range(1, len(value_label_freq)):
        if (value_label_freq[i-1][1][0] > 0 and value_label_freq[i-1][1][1] > 0) or (value_label_freq[i][1][0] > 0 and value_label_freq[i][1][1]>0):
            thresholds.append((value_label_freq[i - 1][0] + value_label_freq[i][0]) / 2)
        elif (value_label_freq[i-1][1][0] > 0 and value_label_freq[i][1][0] == 0) or (value_label_freq[i-1][1][1] > 0 and value_label_freq[i][1][1]==0):
            thresholds.append((value_label_freq[i - 1][0] + value_label_freq[i][0]) / 2)
    return thresholds

def find_continuous_entropy(val_label_freq, thresholds, option4):
    """
    Find best threshold to split on for a continuous feature using IG or IGR
    :param val_label_freq: number of occurences of each value of a given attribute and label pair
    :param thresholds: List of threshold values for a given attribute
    :param option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 
    :return:    best_entropy: value of smallest entropy calculation
                best_split: split value that yeilded this entropy    
    """""
    best_entropy = 100
    best_split = 0

    bins, total_pos, total_neg = compute_threshold_bins(val_label_freq, thresholds)
    leq_pos = 0
    leq_neg = 0

    for bin_ind in range(len(bins)):
        for i in range(len(bins[bin_ind])):
            cur_bin = bins[bin_ind]
            leq_pos += cur_bin[i][1][0]
            leq_neg += cur_bin[i][1][1]
            gr_pos = total_pos - leq_pos
            gr_neg = total_neg - leq_neg
            freq_list = [[leq_pos, leq_neg], [gr_pos, gr_neg]]
            if option4 == 0:
                new_entropy = IG(freq_list)
            else:
                new_entropy = IGR(freq_list)
            if new_entropy < best_entropy:
                best_entropy = new_entropy
                best_split = thresholds[bin_ind]
    return best_entropy, best_split

def most_common_label(examples):
    """
    Compute most common label of examples
    Used to classify a leaf node
    :param examples: examples to classify
    :return: Most common label of examples
    """""

    sum = 0
    for example in examples:  # Get total classifications of 1
        sum += example[-1]
    if sum > examples.__len__() / 2:  # If number of 1's is more than half of all classifications, it is majority
        label = 1
    else:
        label = 0
    return label


def nominal_attribute(key, examples):
    """
    Get the number of positive and negative occurences of each nominal value for an attribute
    :param key: index of attribute
    :param examples: examples to classify by attribute-label pair
    :return: freq: list of list of number of positive and negative occurences of each value of the attribute
    """""
    ex_vals = examples[:, key]
    feat_vals = np.unique(ex_vals)
    # feat_vals = cur_att.values
    freq = []
    for val in feat_vals:
        exs = [ex for ex in examples if ex[key] == val]
        num_p = len([ex for ex in exs if ex[-1] == 1])
        num_n = len([ex for ex in exs if ex[-1] == 0])
        freq.append([num_p, num_n])
    return freq


def output(accuracy, size, maximum_depth, first_feature, fold=None):
    """
    Print output of classification by decision tree
    :param accuracy = fraction of examples correctly classified in testing iteration
    :param size: Number of nodes in tree
    :param max_depth: length of longest sequence of tests from root to leaf
    :param first_feature: name of first feature used to partition data
    :param fold: index of fold for cross validation, None if not using cv 
    """""
    if fold is None:
        print "\nAccuracy: %f\n\nSize: %i\n\nMaximum Depth: %i\n\nFirst Feature: <%s>\n" % (
        accuracy, size, maximum_depth, first_feature)
    else:
        print "\nFold: %i\n\nAccuracy: %f\n\nSize: %i\n\nMaximum Depth: %i\n\nFirst Feature: <%s>\n" % (
        fold, accuracy, size, maximum_depth, first_feature)


def test_tree(tree, test_set):
    """
    Iterate examples in an example set down a tree 
    :param tree: decision tree constructed with ID3, type Node
    :param test_set: dataset to classify by tree
    :return: fraction of correct classifications by tree 
    """""
    correct = 0.0
    total = test_set.__len__()
    for example in test_set:
        label = example.__getitem__(-1)
        tree_label = tree.test_example(example)
        if label == tree_label:
            correct += 1
    return correct / total

