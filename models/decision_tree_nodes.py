# models/decision_tree_nodes.py

class Node:
    """
    A class representing a single node in a decision tree.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Initializes a Node object.
        If it's a leaf node, `value` will be set.
        Otherwise, it's a decision node with `feature` and `threshold`.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Checks if the node is a leaf node."""
        return self.value is not None