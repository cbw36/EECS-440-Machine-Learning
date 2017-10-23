import numpy as np
from mldata import *
from Node import *
from AttributeHelperFunctions import *
from GainCalculations import *



def ID3(examples,attributes,option3,option4, root):
    """
    Decision tree algorithm to produce a tree to classify data based on feature values 
    If all examples positive: return single node tree, label = +
    If all examples negative: return single node tree, label = -
    If attributes empty, return single node tree root, label = most common label
    Otherwise begin:
       A <- attribute from Attributes that best classifies examples (IG or IGR)
       Decision Attribute for Root <- A
       For each value vi of A
           Add new branch below Root corresponding to test A = vi
           Let examples_vi be subset of examples with val vi for A
           If examples_vi empty
               Below branch add leaf with label = most common val of target_attribute in examples
           Else
               Below branch add subtree ID3(examples_vi, target_attributes, Attributes - {A})

    :param examples: dataset of type np.array 
    :param attributes: dict- key: index of attribute, --> value: attribute of type Feature
    :param option3: if nonnegative, set maximum depth of tree.  If 0, grow full tree
    :param option4: 0/1 option to determine split criterion.  If 0, use information gain, if 1 use information gain ratio 
    :param root: root of tree to be created of type Node 
    :return root: fully grown tree from root with dataset examples and features attributes
    """""
    # Base cases
    sum = 0
    for example in examples:
        sum += example[-1]
    if sum == 0:  # if all examples negative
        root.set_label(0)
        return root
    elif sum == examples.__len__():  # if all examples positive
        root.set_label(1)
        return root
    elif len(attributes) == 0:  # if attributes empty
        root.set_label(most_common_label(examples))
        return root

    # Iterative loop
    else:
        best_attribute_dict, best_split = find_best_attribute(examples, attributes, option4)  # Find index of best attribute using IG or IGR
        best_attribute = best_attribute_dict.values()[0]
        att_ind = best_attribute_dict.keys()[0]
        root.set_attribute(best_attribute_dict)
        root.set_split(best_split)
        if best_attribute.type == 'CONTINUOUS':  # Build tree for continuous case
            for i in range(2):
                cur_val = best_split
                if i == 0:
                    examples_val = list(ex for ex in examples if ex[att_ind] <= cur_val)
                    examples_val = np.array(examples_val)
                else:
                    examples_val = list(ex for ex in examples if ex[att_ind] > cur_val)
                    examples_val = np.array(examples_val)
                    cur_val = 'gr'

                if len(examples_val) == 0:
                    label = most_common_label(examples)
                    child = Node(parent=root, label=label, depth=root.get_depth() + 1)  # Construct leaf with majority label
                elif option3 == 0 or option3 > (root.get_depth()+1):  # Maximum depth check
                    child=Node(parent=root, depth=root.get_depth()+1)
                    child = ID3(examples_val, attributes, option3, option4, child)
                else:
                    label = most_common_label(examples_val)
                    child = Node(parent=root, label=label, depth=root.get_depth()+1) # Construct leaf
                root.add_child(cur_val, child)
        else:  # Build for nominal case
            attribute_vals = np.unique(examples[:,att_ind])
            new_attributes = create_new_attribute_list(attributes, att_ind)
            for cur_val in attribute_vals:  # Iterate through unique attribute values
                examples_val = list(ex for ex in examples if ex[att_ind] == cur_val)  # Get a list of all examples with this val NEED TO BE OVER ALL EXAMPLES NOT ITERATIVELY REDUCED.  DO CASES FOR B/C
                examples_val = np.array(examples_val)
                if len(examples_val) == 0:  # If no examples have this value determine majority at root and assume it still holds after sorting by val
                    label = most_common_label(examples)
                    child = Node(parent=root, label=label, depth=root.get_depth() + 1)  # Construct leaf with majority label
                elif option3 == 0 or option3 > (root.get_depth()+1):  # Maximum depth check
                    child = Node(parent=root, depth=root.get_depth()+1)
                    child = ID3(examples_val, new_attributes, option3, option4, child)
                else:
                    label = most_common_label(examples_val)
                    child = Node(parent=root, label=label, depth=root.get_depth()+1)  # Make leaf
                root.add_child(cur_val, child)
    return root