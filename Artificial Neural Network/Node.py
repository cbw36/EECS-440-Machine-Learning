import random
import math

random.seed(12345)


class Node:

    def __init__(self):
        self.child_weights = dict()  # Key is child node, value is weight for that child
        self.net_input = 0  # Total inputs received from parents
        self.activated_input = 0  # value of net_input after activated using sigmoid function
        self.parent_dl_dw = dict()  # Key is parent node, value is dl_dw of that edge
        self.children_sum = 0  # Sum of downstream dl_dws * W/X used for calculating dl_dw

    def add_parent(self, parent):  # Please do not call this
        self.parent_dl_dw[parent] = 0

    def add_child(self, child):  # Automatically calls add_parent
        self.child_weights[child] = random.uniform(-0.1, 0.1)
        child.add_parent(self)

    def update_weight(self, child, delta_weight, learning_rate):  # change the weight for a child
        new_weight = self.__get_weight__(child) - delta_weight*learning_rate
        self.__set_weight__(child, new_weight)

    def increase_net_input(self, new_input, parent=None):  # Take new input and add to net_input
        self.net_input += new_input

    def add_to_child(self, child, input):
        child.increase_net_input(input, self)

    def reset_node(self):  # Reset the node for next test
        self.net_input = 0
        self.activated_input = 0

    def activate_input(self):  # Get value of activated input
        if len(self.parent_dl_dw.keys()) == 0:  # Input nodes don't get activated
            self.activated_input = self.net_input
        else:
            self.activated_input = self.sigmoid_function(self.net_input)

    def __get_activated_input__(self):
        self.activate_input()
        return self.activated_input

    def __set_weight__(self, child, new_weight):
        self.child_weights[child] = new_weight

    def __get_weight__(self, child):
        return self.child_weights[child]

    def __set_dl_dw__(self, parent, dl_dw):
        self.parent_dl_dw[parent] = dl_dw

    def __get_dl_dw__(self, parent):
        return self.parent_dl_dw[parent]

    def __get_children_sum__(self):
        return self.children_sum

    def compute_downstream_sum(self):
        """"
        Compute the downstream contribution towards dl/dw for any node over all nodes closer to output than it
        downstream = dl/dw * weight/X
        """""
        sum = 0
        x = self.__get_activated_input__()
        for child, weight in self.child_weights.items():  # If no children, dont enter loop, and return downstream = 0
            dl_dw = child.parent_dl_dw[self]
            sum += dl_dw * weight/x
        self.children_sum = sum

    def update_dl_dw_hidden(self, parent):  # Updates the dl_dw for a parent
        dl_dw = self.activated_input * (1 - self.activated_input) * parent.__get_activated_input__() * self.__get_children_sum__()
        self.__set_dl_dw__(parent, dl_dw)

    def update_dl_dw_output(self, parent, loss):  # Does ugly calculation
        dl_dw = self.activated_input * (1 - self.activated_input) * parent.__get_activated_input__()* loss
        self.__set_dl_dw__(parent, dl_dw)

    def back_propagate(self, example_class_label, weight_decay):
        self.compute_downstream_sum()
        prediction_error = self.activated_input - example_class_label
        loss = prediction_error + weight_decay
        for parent in self.parent_dl_dw.keys():
            if len(self.parent_dl_dw) != 0:
                if len(self.child_weights) == 0:  # Output layer
                    self.update_dl_dw_output(parent, loss)
                else:
                    self.update_dl_dw_hidden(parent)  # Hidden layer

    def sigmoid_function(self, x):
        e_x = math.exp(-x)
        return 1/(1+e_x)

    def forward_propagate(self):
        self.activate_input()
        for child, weight in self.child_weights.items():
            prop_val = weight * self.activated_input
            child.increase_net_input(prop_val, self)

    def update_weights(self, learning_rate):
        for child, weight in self.child_weights.items():  # and each outgoing weight on the node
            self.update_weight(child, child.__get_dl_dw__(self), learning_rate)
