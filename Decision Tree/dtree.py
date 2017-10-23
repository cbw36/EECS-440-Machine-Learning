import numpy as np
from mldata import *
from Node import *
from ID3 import *
from AttributeHelperFunctions import *
from GainCalculations import *
from CallID3 import *


def dtree(option1, option2, option3, option4):
    """"
    Main Decision tree function
    
    :param option1: path to data
    :param option2: 0/1 option.  If 0, use cross validation.  If 1, run algorithm on full sample
    :param option3: if nonnegative, set maximum depth of tree.  If 0, grow full tree
    :param option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 
    
    Output: accuracy = fraction of examples correctly classified in testing iteration
            size: Number of nodes in tree
            max_depth: length of longest sequence of tests from root to leaf
            first_feature: name of first feature used to partition data
    """""

    exset = parse_c45(option1, '..')  # Data of object ExampleSet
    attributes = dict()  # key: attribute index -- value:Feature object
    for i in range(1, len(exset.schema.features)-1):
        attributes[i] = exset.schema.features[i]
    exset_ar = np.array(exset.to_float())

    # Option 2, test on whole set or cross validation
    if option2 == 0:
        accuracy, size, max_depth, first_feature = run_cv_id3(exset, attributes, option3, option4)
        output(accuracy, size, max_depth, first_feature)
    else:
        accuracy, size, max_depth, first_feature = run_id3(exset_ar, attributes, option3, option4)
        output(accuracy, size, max_depth, first_feature)
    return accuracy


def parse_command_line_inputs():
    """
    Process command line input and call dtree
    option1: path to data
    option2: 0/1 option.  If 0, use cross validation.  If 1, run algorithm on full sample
    option3: if nonnegative, set maximum depth of tree.  If 0, grow full tree
    option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 
    """""
    if len(sys.argv) is not 5:
        raise ValueError('Must call dtree with 4 inputs')
    option1 = sys.argv[1]
    option2 = int(sys.argv[2])
    option3 = int(sys.argv[3])
    option4 = int(sys.argv[4])

    return dtree(option1, option2, option3, option4)

dtree = parse_command_line_inputs()

