class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
    
def insert(root, key):
    if root is None:
        return TreeNode(key)
    else:
        if key < root.key:
            root.left = insert(root.left, key)
        else:
            root.right = insert(root.right, key)
    return root