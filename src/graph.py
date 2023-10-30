import networkx as nx
import matplotlib.pyplot as plt

# Define the nodes and edges based on the provided structure
nodes = ["Age", "Hypertension", "Heart Disease", "Stroke", "Gender",
         "Smoking Status", "Ever Married", "Work Type",
         "Residence Type", "Lifestyle",
         "Average Glucose Level", "BMI"]

edges = [
    ("Age", "Hypertension"),
    ("Age", "Heart Disease"),
    ("Age", "Stroke"),
    ("Gender", "Smoking Status"),
    ("Gender", "Stroke"),
    ("Hypertension", "Stroke"),
    ("Heart Disease", "Stroke"),
    ("Ever Married", "Stroke"),
    ("Work Type", "Stroke"),
    ("Residence Type", "Lifestyle"),
    ("Lifestyle", "Stroke"),
    ("Average Glucose Level", "Stroke"),
    ("BMI", "Hypertension"),
    ("BMI", "Heart Disease"),
    ("BMI", "Stroke"),
    ("Smoking Status", "Hypertension"),
    ("Smoking Status", "Heart Disease"),
    ("Smoking Status", "Stroke")
]

# Create a directed graph using NetworkX
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Define layers for the Bayesian network
layers = {
    0: ["Age", "Gender", "Ever Married", "Work Type", "Residence Type", "Average Glucose Level", "BMI"],
    1: ["Hypertension", "Heart Disease", "Smoking Status", "Lifestyle"],
    2: ["Stroke"]
}


# Define a function to create positions for nodes based on layers
def layered_positions(layers):
    position = {}
    for layer, nodes in layers.items():
        num_nodes = len(nodes)
        for idx, node in enumerate(nodes):
            x = idx - num_nodes / 2 + 0.5
            y = -layer
            position[node] = (x, y)
    return position


# Create positions for nodes using the defined layers
pos = layered_positions(layers)

# Draw the Bayesian network using the positions
plt.figure(figsize=(18, 10))
nx.draw(G, pos, with_labels=True, node_size=4000, node_color="skyblue", font_size=12, width=2, alpha=0.6,
        edge_color="gray")
plt.title("Bayesian Network for Stroke Prediction")
plt.show()
