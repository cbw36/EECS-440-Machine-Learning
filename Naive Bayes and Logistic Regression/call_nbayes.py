from mldata import *
import numpy as np
import random


def cv_nbayes(exset, num_bins, m_estimate, attributes):
    folds = 5
    accuracy_list = []
    precision_list = []
    recall_list = []
    confidences_and_labels = []
    output_lists = [accuracy_list, precision_list, recall_list]
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

        conditional_probabilities = calculate_conditional_probabilities(train_exset, num_bins, m_estimate, attributes)

        accuracy, precision, recall, conf_and_label = run_nbayes(test_exset, conditional_probabilities)

        output_lists[0].append(accuracy)
        output_lists[1].append(precision)
        output_lists[2].append(recall)
        confidences_and_labels.extend(conf_and_label)

    np_output_lists = np.array(output_lists)
    means, stds = calc_mean_sd(np_output_lists)
    aroc = calculate_aroc(confidences_and_labels)
    cv_output(means, stds, aroc)

def full_nbayes(exset_ar, num_bins, m_estimate, attributes):
    conditional_probabilities = calculate_conditional_probabilities(exset_ar, num_bins, m_estimate, attributes)
    # Test tree
    accuracy, precision, recall, conf_and_label = run_nbayes(exset_ar, conditional_probabilities)
    full_output(accuracy, precision, recall, conf_and_label)

def run_nbayes(exset, conditional_probabilities):
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    confidences_and_labels = []

    for ex in exset:
        true_label = ex[-1]
        predicted_label, predicted_confidence = iterate_example(ex, conditional_probabilities)
        confidences_and_labels.append([predicted_confidence, true_label])
        if true_label and predicted_label:
            tp += 1
        elif true_label and (not predicted_label):
            fn += 1
        elif (not true_label) and predicted_label:
            fp += 1
        else:
            tn += 1

    accuracy = (0.0 + tp + tn) / (tp + tn + fp + fn)
    if (tp == 0 and fp == 0):
        precision = 0
    else:
        precision = (0.0 + tp) / (tp + fp)
    if (tp == 0 and fn == 0):
        recall = 0
    else:
        recall = (0.0 + tp) / (tp + fn)
    return accuracy, precision, recall, confidences_and_labels

def iterate_example(ex, conditional_probabilities):
    # Todo update dictionary calls!!!!!!!!
    probx_y = conditional_probabilities['y']
    probx_noty = 1 - probx_y
    for feat in ex:
        pxy = conditional_probabilities(feat)[0]
        px_noty = conditional_probabilities(feat)[1]
        probx_y = probx_y * pxy
        probx_noty = probx_noty * px_noty

    if probx_y > probx_noty:
        return True
    else:
        return False

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

# TODO: Populate
def calculate_conditional_probabilities(exset, num_bins, m_estimate, attributes):
    return 0

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

def cv_output(means, stds, aroc):
    print "\nAccuracy: %f %f\n\nPrecision: %f %f\n\nRecall: %f %f\n\nArea under ROC: %f\n" % (
        means[0], stds[0], means[1], stds[1], means[1], stds[1], aroc)

def full_output(accuracy, precision, recall, confidences_and_labels):  # Output for when CV = 1
    area_under_roc = calculate_aroc(confidences_and_labels)
    print "\nAccuracy: %f\n\nPrecision: %f\n\nRecall: %f\n\nArea under ROC: %f\n" % (
        accuracy, precision, recall, area_under_roc)

def calc_mean_sd(list_of_values):
    sd = list_of_values.std(axis=1)
    mean = list_of_values.mean(axis=1)
    return mean, sd
