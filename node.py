class Node:
  def __init__(self, is_leaf, attributes):
    self.is_leaf = is_leaf
    self.attributes = attributes
    self.label = None

  def teste(self):
    print(self.attributes)
