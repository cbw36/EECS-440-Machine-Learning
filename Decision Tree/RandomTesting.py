#NOT USED IN PROGRAM
from mldata import *
from dtree import *
from AttributeHelperFunctions import *
import numpy as np
from Node import *
import time
from GainCalculations import *

np.set_printoptions(threshold=np.nan)


def test_parse_c45(option1):
    exset = parse_c45(option1, '..')
    class_feature = exset.schema.__getitem__(-1)
    print class_feature
    for example in exset:
        print example
        print example
        break


def test_find_best_attribute():
    for name in ['voting', 'volcanoes', 'spam']:
        exset = parse_c45(name, '..')
        attributes = dict()
        for i in range(1, len(exset.schema)-1):
            attributes[i] = exset.schema.features[i]
        exset_ar = np.array(exset.to_float())
        for option in [0, 1]:
            best_attribute = find_best_attribute(exset_ar, attributes, option)
            print name, option, best_attribute

#test_find_best_attribute()


def test_cv_sorting(option1,num_folds):
    exset = parse_c45(option1, '..')
    (sorted_exset, split_locations) = cv_sorting(exset, num_folds)
    cur_exset = ExampleSet(exset.schema)
    for i in range(split_locations[0], split_locations[1]):
        cur_exset.append(sorted_exset[i])
    print cur_exset

#test_cv_sorting('voting',5)


def create_node(parent):
    root = Node(parent = parent)
    print root.get_size()

#create_node(None)


def test_output(accuracy, size, max_depth, first_feature):
    output(accuracy, size, max_depth, first_feature)

# test_output(0.98,10,18,"Location")


def test_cross_validation(option1, option2, option3, option4):
    start = time.clock()
    dtree(option1, option2, option3, option4)
    end = time.clock()
    print (end-start)

#test_cross_validation('volcanoes', 0, 0, 1)


def test_nominal_attribute(option1,feature_id):
    exset = parse_c45(option1, '..')
    np_exset = np.array(exset.to_float())
    print nominal_attribute(feature_id, np_exset)

#test_nominal_attribute('voting',2)


def question_c():
    #dataset = [('volcanoes', 1), ('spam', 10)]
    dataset = [('spam', 1)]
    output_list = []
    for data in dataset:
        depth_output = []
        data_output = []
        option1, iter_depth = data
        for i in range(1,4):
            max_depth = iter_depth*i
            accuracy = dtree(option1, 0, max_depth, 1)
            depth_output.append(max_depth)
            data_output.append(accuracy)
        output_list.append((depth_output, data_output))
    print output_list

#question_c()


def question_e():
    output = []
    for option1 in ['voting', 'volcanoes', 'spam']:
        cv_or_not = []
        for option2 in [0, 1]:
            accuracy1 = dtree(option1, option2, 1, 0)
            accuracy2 = dtree(option1, option2, 2, 0)
            cv_or_not.append((accuracy1, accuracy2))
        output.append(cv_or_not)
    print output

#question_e()


# Basic running of values
#tree = dtree('example', 0, 0, 0)
#tree = dtree('volcanoes', 1, 0, 0)
#tree = dtree('spam', 0, 0, 0)
