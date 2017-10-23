from Node import *

class Tree:

    def __init__(self, num_nodes_per_layer):
        self.num_nodes_list = num_nodes_per_layer  # List of number of units in that layer
        self.layers = dict()  # Key: layer num (indexing starts at 1), value: list of nodes in layer
        self.num_layers = len(num_nodes_per_layer)
        for index in range(0, self.num_layers):  # For each layer
            self.layers[index+1] = list()  # Make a list, input layer is layer 1 (not 0)
            for i in range(0, num_nodes_per_layer[index]):  # Fill list with new nodes
                new_node = Node()
                if index > 0:  # Check that this is not the first layer
                    for node in self.layers[index]:  # Make fully connected
                        node.add_child(new_node)  # Add parents to new_node and vice versa
                self.layers[index+1].append(new_node)  # Add node to its layer

    def forward_propagation(self):  # Feed example into neural network
        for layer in self.layers.values():  # iterate forward through layers
            for node in layer:  # run on each node in a layer
                node.forward_propagate()

    def backward_propagation(self, example_class_label, weight_decay):  # Calculate new weights
        layers_len = self.num_layers
        for l_ind in range(layers_len-1, 0, -1):
            layer = self.layers.values()[l_ind]
            for node in layer:
                node.back_propagate(example_class_label, weight_decay)

    def update_all_weights(self, learning_rate):  # Update weights after backward propagation
        for i in range(0, len(self.layers)-1):  # iterate forward through layers
            for node in self.layers[i+1]:  # and each node in the layer
                node.update_weights(learning_rate)

    def set_input_nodes(self, example):
        input_layer = self.layers[1]
        for i in range(len(input_layer)):
            input_layer[i].increase_net_input(example[i])

    def iterate_example(self, example, learning_rate, weight_decay_coef):  # Takes an example, fp, bp, updates
        self.set_input_nodes(example[1:-1])
        self.forward_propagation()
        weights_sum = self.compute_all_weights()
        weight_decay = weights_sum * weight_decay_coef
        self.backward_propagation(example[-1], weight_decay)
        self.update_all_weights(learning_rate)
        self.reset_nodes()
        return 0

    def test_example(self, example):
        self.set_input_nodes(example[1:-1])
        self.forward_propagation()
        output_node = self.layers[self.num_layers][0]
        output_value = output_node.__get_activated_input__()
        #print "%f,%f" % (example[-1], output_value)
        if output_value < 0.5:
            output_class = 0
        else:
            output_class = 1
        self.reset_nodes()
        return output_class, output_value

    def compute_all_weights(self):
        weights = 0
        for layer in self.layers.values():
            for node in layer:
                for child, weight in node.child_weights.items():
                    weights += weight
        return weights

    def reset_nodes(self):
        for layer in self.layers.values():
            for node in layer:
                node.reset_node()

    # def compute_all_dl_dws(self):
    #     dl_dws = 0
    #     for layer in self.layers:
    #         for node in layer:
    #             for parent, dl_dw in node.parent_dl_dw.items():
    #                 dl_dws +=  dl_dw
    #     return dl_dws