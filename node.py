class Node:
  def __init__(self):
    self.is_leaf = False
    self.attribute = None
    self.attribute_value = None
    self.label = None
    self.children = []
    self.entropy = None
    self.numeric_value = None
  
  
  
  def print_node(self, level=0):
    val = str(self.attribute_value) if self.attribute_value else ""
    ent = "Entropy: %.3f" % self.entropy
    att = "Leaf" if self.is_leaf else "Best split attribute: " + str(self.attribute)
    lab = "Label: " + str(self.label) if self.label else ""
    return val + " " + ent + "  " + att + " " + lab


