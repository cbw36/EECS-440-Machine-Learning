import numpy as np
from call_nbayes import *
from mldata import *



def nbayes(data_path, cv, num_bins, m_estimate):

    exset = parse_c45(data_path, '..')
    attributes = dict()  # key: attribute index -- value:Feature object
    for i in range(1, len(exset.schema.features) - 1):
        attributes[i] = exset.schema.features[i]
    exset_ar = np.array(exset.to_float())


    if cv == 0:
        cv_nbayes(exset, num_bins, m_estimate, attributes)
    else:
        full_nbayes(exset_ar, num_bins, m_estimate, attributes)


def parse_command_line_inputs():
    """
    Process command line input and call nbayes
    option1: path to data
    option2: 0/1 option.  If 0, use cross validation.  If 1, run algorithm on full sample
    option3: number of bins for continuous features
    option3: value of m-estimate.  If negative, use Laplace smoothing.  If m=0, use MLE estimate.  p is fixed to 1/v for variable with v values

    """""
    if len(sys.argv) is not 4:
        raise ValueError('Must call nbayes with 3 inputs')
    option1 = sys.argv[1]
    option2 = int(sys.argv[2])
    option3 = int(sys.argv[3])
    option4 = int(sys.argv[4])

    return nbayes(option1, option2, option3, option4)

nbayes = parse_command_line_inputs()