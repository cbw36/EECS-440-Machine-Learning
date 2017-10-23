import numpy as np
from mldata import *
from Node import *
from AttributeHelperFunctions import *
import random


def IG(list_of_frequency):
    information_gain = entropy(list_of_frequency)
    #print list_of_frequency
    return information_gain


def IGR(list_of_frequency):
    information_gain = entropy(list_of_frequency)
    pos_count = 0
    neg_count = 0
    for frequency in list_of_frequency:
        [pos, neg] = frequency
        pos_count += pos
        neg_count += neg
    feature_entropy = entropy([[pos_count, neg_count]])
    information_gain_ratio = information_gain/feature_entropy
    return information_gain_ratio

def entropy(list_of_frequency):
    entropy = 0
    total_number = float(sum(e[0]+e[1] for e in list_of_frequency))
    for att_split in list_of_frequency:
        total_attribute = float(att_split[0] + att_split[1])
        if total_attribute == 0 or total_number == 0:
            pass
        else:
            att_fraction = total_attribute/total_number
            for att_class in att_split:
                class_fraction = att_class/total_attribute
                if class_fraction == 0:
                    pass
                else:
                    entropy = entropy - att_fraction * (class_fraction*np.log2(class_fraction))
    return entropy
