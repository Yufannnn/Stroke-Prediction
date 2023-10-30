import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Load the saved Bayesian Network
with open("saved_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

for cpd in loaded_model.get_cpds():
    print(cpd)
    print("-----\n")

# Convert Bayesian Model to Directed Graph
G = nx.DiGraph()
G.add_edges_from(loaded_model.edges())

# Plot the Directed Graph
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=15, width=2, alpha=0.6)
plt.title("Bayesian Network")
plt.show()

# Display the CPDs for specific nodes
print(loaded_model.get_cpds("stroke"))

