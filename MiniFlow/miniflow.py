import numpy as np

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this NOde receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound Node here, add this Node as an
        # outbound Node to that Node

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        # A calculated value
        self.value = None

        #Gradients dict, keys are the inputs to this node and 
        #the values are the partial derivative of the node 
        #with respect to the input
        self.gradients = {}

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on inbound_nodes and store the result in
        self.value.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every node that uses this class as a base class will need to define its
        own 'backward' method.
        """

        raise NotImplementedError

class Input(Node):
    def __init__(self):
        # An INput node has no inbound nodes,
        # so no need to pass anything to the Node
        # instantiator
        Node.__init__(self)

    # Note: INput node is the only node where the value
    # may be passed as an argument to forward()
    #
    # All other node implementations should get the value
    # of the previous nodes from self.inbound_nodes
    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        # An  input node has no inputs so the gradient (derivative)
        # is zero.
        # The key, 'self', is reference to this object.
        self.gradients = {self:0}
        # Weights and biases may be inputs so you need to sum
        # the gradient from output gradients
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x,y])

    def forward(self):
        total = 0 
        for node in self.inbound_nodes:
            total += node.value
        self.value = total

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [input,weights,bias])

        # Note: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stoed within the 
        #respective nodes.

    def forward(self):
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        self.value = np.dot(inputs,weights) + bias)

    def backward(self):
        """
        Caculates the gradient based on the output values.
        """
        #Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) 
                             for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost,
                    self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights
            self.gradients[self.inbound_nodes[1]] += np.dot(grad_cost,
                    self.inbound_nodes[0].value.T)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost,
                    axis=0,keepdims=False)



class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from forward because it will be used later with
        backward as well.
        
        Return the result of the sigmoid function
        """
        return 1/(1+np.exp(-x))

    def forward(self):
        """
        Set the value of this node to the result of the sigmoid function,
        '_sigmoid'.
        """
        X = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)

class MSE(Node):
    def __init__(self, y, a):
        """
        the mean squared error cost function.
        Should be used as the last node for a network.
        """
        Node.__init__(self, [y,a])

    def forward(self):
        """
        Calculates the mean squared error.
        """
        y = self.inbound_nodes[0].value.reshape(-1,1)
        a = self.inbound)nodes[1].value.reshape(-1,1)
        m = len(y)
        self.value = np.mean(np.square(y - a)) 


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    'feed_dict': A dictionary where the key is a 'Input' node and the value is
    the respective value feed to that node

    Returns a list of sorted nodes
    """

    input_nodes = [n for  n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            #if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(graph):
    """
    Performs a forward pass and backwardpass through a list of sorted nodes.

    Arguments:
        'graph': A topologically sorted list of nodes.

    Returns the output Node's value
    """
    
    #forward pass
    for n in graph:
        n.forward()

    #Backward pass
    for n in graph[::-1]:
        n.backward()

    return output_node.value


