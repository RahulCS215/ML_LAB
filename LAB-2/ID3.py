import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy
import graphviz

# Load dataset
dataset = pd.read_csv('/content/Tennis (1).csv')  # Update this path if necessary
X = dataset.values
attribute = ['Outlook', 'Temp', 'Humidity', 'Wind']

# Node class for tree structure
class Node:
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = []

# Function to calculate entropy
def findEntropy(data, rows):
    yes, no = 0, 0
    idx = len(data[0]) - 1
    entropy = 0

    for i in rows:
        if data[i][idx] == 'Yes':
            yes += 1
        else:
            no += 1

    if yes == 0 or no == 0:
        return 0, 1 if yes > 0 else 0

    x = yes / (yes + no)
    y = no / (yes + no)
    entropy = - (x * math.log2(x) + y * math.log2(y))

    return entropy, -1

# Function to calculate information gain
def findMaxGain(data, rows, columns):
    maxGain, retidx = 0, -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGain, retidx, ans

    for j in columns:
        mydict = {}
        for i in rows:
            key = data[i][j]
            mydict[key] = mydict.get(key, 0) + 1

        gain = entropy
        for key in mydict:
            yes, no = 0, 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Yes':
                        yes += 1
                    else:
                        no += 1
            x = yes / (yes + no) if yes + no > 0 else 0
            y = no / (yes + no) if yes + no > 0 else 0
            if x > 0 and y > 0:
                gain -= (mydict[key] / len(rows)) * (x * math.log2(x) + y * math.log2(y))

        if gain > maxGain:
            maxGain, retidx = gain, j

    return maxGain, retidx, ans

# Function to build the decision tree
def buildTree(data, rows, columns):
    maxGain, idx, ans = findMaxGain(data, rows, columns)
    root = Node()
    if maxGain == 0:
        root.value = 'Yes' if ans == 1 else 'No'
        return root

    root.value = attribute[idx]
    mydict = {data[i][idx]: [] for i in rows}
    for i in rows:
        mydict[data[i][idx]].append(i)

    new_columns = [col for col in columns if col != idx]
    for key in mydict:
        child = buildTree(data, mydict[key], new_columns)
        child.decision = key
        root.childs.append(child)
    return root

# Function to visualize tree using Graphviz
def visualize_tree(root):
    dot = graphviz.Digraph(format='png')
    
    def add_nodes_edges(node, parent_name="Root"):
        if node:
            node_name = f"{node.decision}\n{node.value}"
            dot.node(node_name, label=node.value if node.decision is None else f"{node.decision}\n{node.value}")
            if parent_name != "Root":
                dot.edge(parent_name, node_name)

            for child in node.childs:
                add_nodes_edges(child, node_name)
    
    add_nodes_edges(root)
    dot.render('decision_tree', format='png', view=True)  # Saves and opens the tree

# Run the ID3 algorithm and visualize
def calculate():
    rows = list(range(len(X)))
    columns = list(range(len(attribute)))
    root = buildTree(X, rows, columns)
    visualize_tree(root)

calculate()
