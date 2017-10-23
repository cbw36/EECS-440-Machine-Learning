import numpy as np
from mldata import *
from call_ann import *

def ann(data_path, cv, num_hidden, weight_decay_coef, num_iter):
    """"
    Process command line input and call dtree
    data_path: path to data
    cv: 0/1 option.  If 0, use cross validation.  If 1, run algorithm on full sample
    num_hidden: int to set number of hidden units
    weight_decay_coef: Float to set value of weight decay coefficient
    num_iter: integer that sets number of training iterations.  If 0 or negative, train until convergence
    """""

    exset = parse_c45(data_path, '..')
    exset_ar = np.array(exset.to_float())
    normalization_data = calculate_mean(exset_ar), calculate_sd(exset_ar)
    if cv == 0:
        cv_ann(exset, num_hidden, weight_decay_coef, num_iter, normalization_data)
    else:
        full_ann(exset_ar, num_hidden, weight_decay_coef, num_iter, normalization_data)


def parse_command_line_inputs():
    """
    Process command line input and call dtree
    option1: path to data
    option2: 0/1 option.  If 0, use cross validation.  If 1, run algorithm on full sample
    option3: int to set number of hidden units
    option4: Float to set value of weight decay coefficient
    option5: integer that sets number of training iterations.  If 0 or negative, train until convergence
    """""
    if len(sys.argv) is not 6:
        raise ValueError('Must call dtree with 5 inputs')
    option1 = sys.argv[1]
    option2 = int(sys.argv[2])
    option3 = int(sys.argv[3])
    option4 = float(sys.argv[4])
    option5 = int(sys.argv[5])

    return ann(option1, option2, option3, option4, option5)

ann = parse_command_line_inputs()