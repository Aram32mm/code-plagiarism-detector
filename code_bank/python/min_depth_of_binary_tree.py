def minDepth(self, root):
    if not root: return 0
    d = map(self.minDepth, (root.left, root.right))
    return 1 + (min(d) or max(d))

def minDepth(self, root):
    if not root: return 0
    d, D = sorted(map(self.minDepth, (root.left, root.right)))
    return 1 + (d or D)