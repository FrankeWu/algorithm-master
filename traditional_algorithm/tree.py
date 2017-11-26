"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
class BinaryTreeNode:
    """"""
    def __init__(self, left_node=None, right_node=None):

        self.left = left_node
        self.right = right_node


class BinaryTree:
    """"""
    def __init__(self, tree_root=None):

        self.root = tree_root

    def is_null_tree(self):
        """Whether tree is null tree."""
        return self.root == None


if __name__ == "__main__":

    Tree = BinaryTree()
    print(Tree.is_null_tree())

