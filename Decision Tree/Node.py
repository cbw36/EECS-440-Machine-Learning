import numpy as np
from mldata import *
from AttributeHelperFunctions import *
from GainCalculations import *



class Node:
    """"
    Class to store nodes of tree.  Decision tree is represented as a set of parent-child relations of objects of type Node
    """""
    def __init__(self, parent=None, children=None, label=None, attribute=None, branches=None, depth=0, split=None):
        """
        Constructor for Node
        :param parent: parent of node, type Node
        :param children: list of all children
        :param label: label tree assigns data if leaf, None if not leaf
        :param attribute: attribute node splits on.  Type dict of {attribute index: Feature}
        :param branches: Dict of {value branch splits on:child}
        :param depth: depth of node
        :param split: Split value if continuous, else None
        """""
        self.parent = parent
        self.children = list()
        self.label = label
        self.attribute = attribute
        self.branches = dict()
        self.depth = depth
        self.split = split

    def add_child(self, branch,child):
        """
        Add child to branches dictionary at its corresponding value and append it to child list
        :param branch: value of root's attribute this child corresponds to 
        :param child: child of self, object Node
        :return: 
        """""
        self.branches[branch] = child
        self.children.append(child)


    def set_label(self, label):
        """
        Set class label of leaf node 
        :param label: label algorithm guesses of any data that reaches this leaf
        """""
        self.label = label

    def set_attribute(self, attribute):
        """
        Set attribute that this node splits on
        :param attribute: Dict- key: feature id -- value:Feature object
        """""
        self.attribute = attribute

    def get_attribute(self):
        """
        Return node's attribute
        :return: attribute dict
        """""
        return self.attribute

    def set_parent(self, parent):
        """
        Set parent Node of node
        :param parent: parent of node of type Node
        :return: 
        """""
        self.parent = parent

    def set_depth(self, val):
        """
        Set depth of node (1+parents depth)
        :param val: depth of node, type int
        """""
        self.depth = val

    def get_depth(self):
        """
        Return depth of node
        :return: depth: int of node's depth
        """""
        return self.depth

    def set_split(self, split):
        """
        Set split value for continuous attribute, None for nominal
        :param split: value to split node's attribute on
        """""
        self.split=split

    def test_example(self, example):
        """
        Iterate an example down tree based on its attributes to detrmine the label the tree assigns it
        :param example: data point to send through tree, type Example
        :return: label tree assigns to example
        """""
        if len(self.children) == 0:  # Determine if this is a leaf node
            return self.label
        elif self.attribute.values()[0].type == 'CONTINUOUS':
            val = example[self.attribute.keys()[0]]
            split_val = self.split
            if val <= split_val:
                next_child = self.children[0]
                return next_child.test_example(example)
            else:
                next_child = self.children[1]
                return next_child.test_example(example)
        else:
            branch = example[self.attribute.keys()[0]]  # Find branch value
            if branch in self.branches.keys():  # Determine if the node has this branch value
                next_child = self.branches[branch]  # Get the next child
                return next_child.test_example(example)  # Go to next child
            else:  # This branch value was not in the example set
                return self.label

    def get_size(self):
        """
        Get number of nodes in tree rooted at Node
        :return: number of nodes, type int
        """""
        if len(self.children) == 0:
            return 1
        total_size = 1
        for child in self.children:
            total_size = total_size + child.get_size()
        return total_size

    def get_max_depth(self):
        """
        Get maximum depth of tree rooted at node
        :return: depth, type int
        """""
        return self.max_depth_inner_function(-1)

    def max_depth_inner_function(self, parent_depth):
        """
        Helper function to compute maximum depth
        :param parent_depth: Depth of parent node, int
        :return: depth of node, int
        """""
        my_depth = parent_depth + 1
        if len(self.children) == 0:
            return my_depth
        max_depth = 0
        for child in self.children:
            max_depth = max(max_depth, child.max_depth_inner_function(my_depth))

        return max_depth