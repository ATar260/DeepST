import community as louvain
import networkx as nx

# Example: Create a sample graph using NetworkX
G = nx.erdos_renyi_graph(30, 0.05)

# Run the Louvain algorithm
partition = louvain.best_partition(G)

# Print the resulting clusters
print(partition)
