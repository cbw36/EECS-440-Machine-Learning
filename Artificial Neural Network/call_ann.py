import numpy as np
from mldata import *
import random
from Tree import *



random.seed(12345)  # Output same random number


def cv_ann(exset, num_hidden, weight_decay_coef, num_iter, normalization_data):
    accuracy = 0.0
    folds = 5
    accuracy_list = []
    precision_list = []
    recall_list = []
    confidences_and_labels= []
    output_lists = [accuracy_list, precision_list, recall_list]
    # Calculating divisions for cross validation
    (sorted_exset, split_locations) = cv_sorting(exset, folds)
    for i in range(folds):  # TODO REMOVED ITERATION OF FOLDS
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
        normalized_train_exset = normalize_data(train_exset, normalization_data)
        test_exset = np.array(test_exset.to_float())
        normalized_test_exset = normalize_data(test_exset, normalization_data)

        # Run ANN on training set
        tree = build_ann(normalized_train_exset, num_hidden, weight_decay_coef, num_iter)

        accuracy, precision, recall, conf_and_label = evaluate_ann(tree, normalized_test_exset)

        output_lists[0].append(accuracy)
        output_lists[1].append(precision)
        output_lists[2].append(recall)
        confidences_and_labels.extend(conf_and_label)

    np_output_lists = np.array(output_lists)
    means, stds = output_calculation(np_output_lists)
    aroc = calculate_aroc(confidences_and_labels)
    cv_output(means, stds, aroc)


def full_ann(exset, num_hidden, weight_decay_coef, num_iter, normalization_data):
    normalized_exset = normalize_data(exset, normalization_data)
    tree = build_ann(normalized_exset, num_hidden, weight_decay_coef, num_iter)
    # Test tree
    accuracy, precision, recall, confidences_and_labels = evaluate_ann(tree, normalized_exset)
    full_output(accuracy, precision, recall, confidences_and_labels)


def build_ann(exset, num_hidden, weight_decay_coef, num_iter):
    learning_rate = 0.01
    num_examples = exset.shape[0]
    num_nodes_l1 = exset.shape[1] - 2
    num_nodes_l3 = 1

    if num_hidden !=0:
        num_nodes_l2 = num_hidden
        num_nodes_per_layer = [num_nodes_l1, num_nodes_l2, num_nodes_l3]
    else:
        num_nodes_per_layer = [num_nodes_l1, num_nodes_l3]
    tree = Tree(num_nodes_per_layer)
    if num_iter < 1:  # train until convergence
        eps = 0.01  # convergence test
        delta_w = 1  # initialize delta larger than eps
        iteration = 0

        while delta_w > eps:
            w_start = tree.compute_all_weights()
            w_final = iterate_example_set(tree, exset, learning_rate, weight_decay_coef)
            iteration += np.shape(exset)[0]
            delta_w = abs(w_start - w_final)
        print iteration

    else:
        iteration = 0
        while iteration < num_iter:
            iterate_example_set(tree, exset, learning_rate, weight_decay_coef)
            iteration += np.shape(exset)[0]
    return tree


def cv_sorting(exset, num_folds):
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


def normalize_data(exset_ar, normalization_data):
    means, std_devs = normalization_data
    normalized_exset = np.divide(np.subtract(exset_ar, means), std_devs)
    return normalized_exset


def calculate_mean(exset_ar):
    means = exset_ar.mean(axis=0, dtype=float)
    means[-1] = 0
    return means


def calculate_sd(exset_ar):
    std_devs = np.std(exset_ar, axis=0)
    std_devs[-1] = 1
    std_devs[std_devs == 0] = 1
    return std_devs


def evaluate_ann(tree, exset):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    confidences_and_labels = []

    for ex in exset:
        true_label = ex[-1]
        predicted_label, predicted_confidence = tree.test_example(ex)
        confidences_and_labels.append([predicted_confidence, true_label])
        if true_label and predicted_label:
            tp += 1
        elif true_label and (not predicted_label):
            fn += 1
        elif (not true_label) and predicted_label:
            fp += 1
        else:
            tn += 1

    accuracy = (0.0 + tp + tn) / (tp+tn+fp+fn)
    if (tp == 0 and fp == 0):
        precision = 0
    else:
        precision = (0.0 + tp)/(tp+fp)
    if (tp == 0 and fn == 0):
        recall = 0
    else:
        recall = (0.0 + tp)/(tp + fn)
    return accuracy, precision, recall, confidences_and_labels


def iterate_example_set(tree, exset, learning_rate, weight_decay_coef):
    for ex in exset:
        tree.iterate_example(ex, learning_rate, weight_decay_coef)
    weights_sum = tree.compute_all_weights()
    return weights_sum


def output_calculation(list_of_values):
    sd = list_of_values.std(axis=1)
    mean = list_of_values.mean(axis=1)
    return mean, sd


def cv_output(means, stds, aroc):
    #accuracy_mean = means[0]
    #accuracy_std = stds[0]
    #accuracy_mean = means[1]
    #accuracy_std = stds[1]
    #accuracy_mean = means[2]
    #accuracy_std = stds[2]
    print "\nAccuracy: %f %f\n\nPrecision: %f %f\n\nRecall: %f %f\n\nArea under ROC: %f\n" % (
        means[0], stds[0], means[1], stds[1], means[1], stds[1], aroc)


def full_output(accuracy, precision, recall, confidences_and_labels):  # Output for when CV = 1
    area_under_roc = calculate_aroc(confidences_and_labels)
    print "\nAccuracy: %f\n\nPrecision: %f\n\nRecall: %f\n\nArea under ROC: %f\n" % (
        accuracy, precision, recall, area_under_roc)


def calculate_aroc(confidences_and_labels):
    tp_rate = []
    fp_rate = []
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # Compute area under ROC
    confidences_and_labels = sorted(confidences_and_labels, reverse=True)
    for conf_and_label in confidences_and_labels:
        label = conf_and_label[1]
        if label:
            fn += 1
        else:
            tn += 1

    for conf_and_label in confidences_and_labels:
        label = conf_and_label[1]
        if label:
            fn -= 1
            tp += 1
        else:
            tn -= 1
            fp += 1
        tp_rate.append((tp + 0.0) / (tp + fn))
        fp_rate.append((fp + 0.0) / (tn + fp))
    aroc = np.trapz(tp_rate, fp_rate)
    return aroc
