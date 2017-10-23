import numpy as np
from mldata import *
import random
from Tree import *
from Node import *
import time
from ann import *
from call_ann import *

################
##Node Testing##
################


def make_node():
    return Node()


def make_parent_child_pair():
    node1 = make_node()
    node2 = make_node()
    node1.add_child(node2)
    return node1, node2


def make_chain():
    node1 = make_node()
    node2 = make_node()
    node3 = make_node()
    node1.add_child(node2)
    node2.add_child(node3)
    return node1, node2, node3


def make_branch():
    node1, node2, node3 = make_chain()
    node4 = make_node()
    node2.add_child(node4)
    return node1, node2, node3, node4


def test_add_child():
    print "Testing Add Child"
    node1, node2 = make_parent_child_pair()
    if node2 not in node1.child_weights.keys():
        print "Error node2 not child of node1"

    if node1 not in node2.parent_values.keys():
        print "Error node1 not parent of node2"

    node2_weight = node1.child_weights[node2]
    if node2_weight < 0 or node2_weight > 1:
        print "Weight, %d, not allowed" % node2_weight

    node1_value = node2.parent_values[node1]
    if node1_value != 0:
        print "Initial value not 0"
    print "Finished Add Child"


def test_set_weight(new_weight):
    print "Testing Set Weight"
    node1, node2 = make_parent_child_pair()
    old_weight = node1.child_weights[node2]
    node1.__set_weight__(node2, new_weight)
    if node1.child_weights[node2] != new_weight:
        print "Weight not setting"
    print "Finished Set Weight"


def test_update_weight(new_weight, delta_weight, learning_rate):
    print "Testing Update Weight"
    node1, node2 = make_parent_child_pair()
    node1.__set_weight__(node2, new_weight)
    node1.update_weight(node2, delta_weight, learning_rate)
    if node1.child_weights[node2] != (new_weight - (delta_weight*learning_rate)):
        print "Incorrect Calculation"
    print "Finished Update Weight"


def test_increase_net_input(new_input1, new_input2):
    print "Testing Increase Net Input"
    node1, node2 = make_parent_child_pair()

    if node2.net_input != 0:
        print "Initial Net Input is not 0"

    node2.increase_net_input(new_input1)

    if node2.net_input != new_input1:
        print "Increase Net Input not incrementing correctly"

    if len(node2.parent_values) != 1:
        print "Parent was incorrectly added"

    node2.increase_net_input(new_input2, node1)

    if node2.net_input != new_input1 + new_input2:
        print "Increase Net Input not incrementing correctly"

    if node2.parent_values[node1] != new_input2:
        print "Parent Values not updating"

    print "Finished Increase Net Input"


def test_add_to_child(value):
    print "Testing Add To Child"
    node1, node2 = make_parent_child_pair()
    node1.add_to_child(node2, value)
    if node2.__get_value__(node1) != value:
        print "Add To Child not incrementing correctly"
    print "Finished Add To Child"


def test_reset_node(value):
    print "Testing Reset Node"
    node1, node2 = make_parent_child_pair()
    node1.add_to_child(node2, value)
    node2.activate_input()
    node2.reset_node()
    if node2.activated_input != 0:
        print "Activated Input not reset"
    if node2.net_input != 0:
        print "Net Input not reset"
    print "Finished Reset Node"


def test_activate_input(value):
    print "Testing Activate Input"
    # No parent
    node1 = make_node()
    if node1.activated_input != 0:
        print "Activated Input initial value incorrect"

    node1.activate_input()
    if node1.activated_input != 0.0:
        print "Activate Input with no parent not setting correct value"

    node1.increase_net_input(value)
    node1.activate_input()
    if node1.activated_input != value:
        print "Activate Input with no parent not setting correct value"

    # With parent
    node1, node2 = make_parent_child_pair()
    node2.activate_input()
    if node2.activated_input != 0.5:
        print "Activate Input not setting correct value"

    node1.add_to_child(node2, value)
    node2.activate_input()
    if node2.activated_input != node1.sigmoid_function(value):
        print "Activate Input not setting correct value"

    print "Finished Activate Input"


def test_get_activated_input(value):
    print "Testing Get Activated Input"
    # No parent
    node1 = make_node()
    if node1.__get_activated_input__() != 0:
        print "Get Activated Input not setting correct value"

    node1.increase_net_input(value)
    if node1.__get_activated_input__() != value:
        print "Get Activated Input not setting correct value"

    # With parent
    node1, node2 = make_parent_child_pair()
    if node2.__get_activated_input__() != 0.5:
        print "Get Activated Input not setting correct value"

    node1.add_to_child(node2, value)
    if node2.__get_activated_input__() != node1.sigmoid_function(value):
        print "Get Activated Input not setting correct value"

    print "Finished Get Activated Input"

# Did not test all getter and setter methods


def test_forward_propagate(input1, input2):
    node1, node2, node3 = make_chain()
    node1.increase_net_input(input1)
    node1.forward_propagate()
    # print node2.parent_values[node1]
    node2.forward_propagate()
    # print node3.parent_values[node2]
    node1, node2, node3, node4 = make_branch()
    node1.increase_net_input(input2)
    node1.forward_propagate()
    # print node2.parent_values[node1]
    node2.forward_propagate()
    # print node3.parent_values[node2]
    # print node4.parent_values[node2]
    node3.forward_propagate()
    node4.forward_propagate()
    return node1, node2, node3, node4


def test_back_propagate(input1, input2, example_label):
    node1, node2, node3, node4 = test_forward_propagate(input1, input2)
    node3.back_propagate(example_label)
    node4.back_propagate(example_label)
    node2.back_propagate(example_label)
    node1.back_propagate(example_label)
    return node1, node2, node3, node4


def test_update_weights(input1, input2, example_label, learning_rate):
    node1, node2, node3, node4 = test_back_propagate(input1, input2, example_label)
    # print node2.child_weights
    node2.update_weights(learning_rate)
    # print node2.child_weights
    # print node1.child_weights
    node1.update_weights(learning_rate)
    # print node1.child_weights


def test_sigmoid_function():
    print "Testing Sigmoid Function"
    node1 = make_node()
    if node1.sigmoid_function(0) != 0.5:
        print "Sigmoid function broken"
    print "Finished Sigmoid Function"


def test_all():
    test_add_child()
    test_set_weight(0.3)
    test_update_weight(0.5, 0.1, 0.1)
    test_increase_net_input(0.3, 0.5)
    test_add_to_child(0.4)
    test_reset_node(0.5)
    test_activate_input(0.7)
    test_get_activated_input(0.8)
    test_sigmoid_function()
    test_update_weights(0.5, 0.7, 1, 0.01)

# test_all()


def test_ann(input1, input2, input3, input4, input5):
    ann(input1, input2, input3, input4, input5)

# test_ann("voting", 0, 3, 0.1, 5)

# Testing Tree Constructor

#ann('voting', cv, num_hidden, weight_decay_coef, num_iter)

def test_tree_constructor(exset, num_hidden):
    num_examples = exset.shape[0]
    num_nodes_l1 = exset.shape[1]
    num_nodes_l2 = num_hidden
    num_nodes_l3 = 1
    num_nodes_per_layer = [num_nodes_l1, num_nodes_l2, num_nodes_l3]
    return Tree(num_nodes_per_layer)


#tree = test_tree_constructor(option1, num_hidden)  # WORKS!


def problem_a():
    print "Voting"
    ann('voting', 0, 0, 0, 0)
    print "\nvolcanoes"
    ann('volcanoes', 0, 0, 0, 0)
    print "\nspam"
    ann('spam', 0, 0, 0, 0)

problem_a()

def problem_b():
    print "PROBLEM B"
    iterations = [3000, 6000, 9000]
    print "Volcanoes"
    for iteration in iterations:
        ann('volcanoes', 0, 5, 0.01, iteration)
    print "\nSpam"
    for iteration in iterations:
        ann('spam', 0, 5, 0.01, iteration)

def problem_c():
    print "PROBLEM C"
    hidden_list = [75, 150, 225]
    for num_hidden in hidden_list:
        num_iter = 100 * num_hidden
        ann('volcanoes', 0, num_hidden, 0.01, num_iter)

#problem_b()
#problem_c()