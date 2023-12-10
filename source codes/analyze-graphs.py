#%%
from audioop import reverse
from ctypes import Union
import networkx as nx
from networkx.algorithms import graph_edit_distance
from neo4j import GraphDatabase
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from matplotlib_venn import venn3
#%%
#Log in to Neo4j
uri      = "bolt://localhost:7687" # in Neo4j Desktop
                              # custom URL for Sandbox or Aura
user     = "neo4j"            # your user name 
                              # default is always "neo4j" 
                              # unless you have changed it. 
password = "pass"
driver = GraphDatabase.driver(uri=uri,auth=(user,password))
driver.verify_connectivity()

#%%
#  GET graph of gpt4
query = """
MATCH (n)-[r]->(c)  where r.annotationDatasource = ["gpt4"] RETURN *
"""
results = driver.session().run(query)
G_gpt4 = nx.MultiGraph()
nodes = list(results.graph()._nodes.values())

for node in nodes:
    G_gpt4.add_node(node.id, labels=node._labels, properties=node._properties)
rels = list(results.graph()._relationships.values())
print(len(rels))

for rel in rels:
    G_gpt4.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

edge_list_gpt4 = list(G_gpt4.edges)
print("build graph of gpt4")
# % Graph of gpt 35
query = """
MATCH (n)-[r]->(c)  where r.annotationDatasource = ["gpt3.5"] RETURN *
"""
results = driver.session().run(query)
G_gpt35 = nx.MultiGraph()
nodes = list(results.graph()._nodes.values())
for node in nodes:
    G_gpt35.add_node(node.id, labels=node._labels, properties=node._properties)
rels = list(results.graph()._relationships.values())
for rel in rels:
    G_gpt35.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

edge_list_gpt35 = list(G_gpt35.edges)
print("build graph of gpt35")
# % Graph of sherpa
query = """
MATCH (n)-[r]->(c)  where r.annotationDatasource = ["sherpa"] RETURN *
"""
results = driver.session().run(query)
G_sherpa = nx.MultiGraph()
nodes = list(results.graph()._nodes.values())
for node in nodes:
    G_sherpa.add_node(node.id, labels=node._labels, properties=node._properties)
rels = list(results.graph()._relationships.values())
for rel in rels:
    G_sherpa.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

edge_list_sherpa = list(G_sherpa.edges)
#print(len(edge_list_sherpa))
print("build graph of sherpa")
# % Graph of tau
query = """
MATCH (n)-[r]->(c)  where r.annotationDatasource = ["tau-subgraph-pharmacome"] RETURN *
"""
results = driver.session().run(query)
G_tau = nx.MultiGraph()
nodes = list(results.graph()._nodes.values())
for node in nodes:
    G_tau.add_node(node.id, labels=node._labels, properties=node._properties)

rels = list(results.graph()._relationships.values())
for rel in rels:
    G_tau.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

edge_list_tau = list(G_tau.edges)
print("build graph of tau-subgraph--pharmacome")
#distance = nx.optimize_graph_edit_distance(G_tau, G_gpt35)

# %% Compute optimized edit distance
response = nx.optimize_edit_paths(G_gpt4, G_sherpa)
print("done optimize path computation")
for item in response:
    print(item)
print("done")

#%% store nodes and edges
print("nodes")
set_tau = G_tau.nodes()
set_sherpa = G_sherpa.nodes()
set_gpt4 = G_gpt4.nodes()
set_gpt35 = G_gpt35.nodes()
print("sherpa",len(set_sherpa))
print("tau",len(set_tau))
print("gpt4",len(set_gpt4))
print("gpt35",len(set_gpt35))
print(set_sherpa)
set_tau_edges = G_tau.edges()
set_sherpa_edges = G_sherpa.edges()
set_gpt4_edges = G_gpt4.edges()
set_gpt35_edges = G_gpt35.edges()

#%% Interssection of rels and print values
result = set(set_tau_edges).intersection(set(set_gpt35_edges))
print(result)
print("node intersection")
result = set(set_tau).intersection(set(set_gpt35))
print(result)
#%%intersection of nodes and print BEL values
set_tau = G_tau.nodes(data = True)
node_id_bel = {}
for item in set_tau:
    node_id_bel[item[0]] = item[1]["properties"]["bel"]
#print(node_id_bel)
set_tau = G_tau.nodes()
result = set(set_tau).intersection(set(set_sherpa).intersection(set(set_gpt35),set(set_gpt4)))
result = set(set_tau).intersection(set(set_sherpa))
for item in result:
    #print(item)
    print(node_id_bel[item])
print(len(result))
#%% nodes (or rels) that appear just in one model
from functools import reduce
A = [set(set_tau_edges), set(set_sherpa_edges), set(set_gpt35_edges), set(set_gpt4_edges)]
B = set()
for i in range(len(A)):
    U = reduce(set.union, A[:i]+A[(i+1):])
    B = B.union(set.difference(A[i], U))
#print(B)
for tuples in B:
    head = tuples[0]
    tail = tuples[1]
    print(G_sherpa.get_edge_data(head,tail))

#%% draw figure venn 4d
from venny4py.venny4py import *
#dict of sets
print("Venn for edges")
sets = {
    'Tau KG': set(list(set_tau_edges)),
    'GPT3.5': set(list(set_gpt35_edges)),
    'GPT4': set(list(set_gpt4_edges)),
    'Sherpa': set(list(set_sherpa_edges))}
    
venny4py(sets=sets)

print("Venn for nodes")
sets = {
    'Tau KG': set(list(set_tau)),
    'GPT3.5': set(list(set_gpt35)),
    'GPT4': set(list(set_gpt4)),
    'Sherpa': set(list(set_sherpa))}
    
venny4py(sets=sets)
#%%
# %% Test intersection of relations of graphs
query = """
MATCH (n)-[r]->(c) RETURN *
"""
results = driver.session().run(query)
G_total = nx.MultiGraph()
nodes = list(results.graph()._nodes.values())
for node in nodes:
    G_total.add_node(node.id, labels=node._labels, properties=node._properties)

rels = list(results.graph()._relationships.values())
for rel in rels:
    G_total.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)

total_edges = list(G_total.edges)
print("build graph")
G_total.edges(data=True)
print(G_total.edges(data=True))



#%%ven 4d for common relations without caring about evidences
from venny4py.venny4py import *
sets = {
    'Tau KG': set(list(tau_edges)),
    'GPT3.5': set(list(gpt35_edges)),
    'GPT4': set(list(gpt4_edges)),
    'Sherpa': set(list(sherpa_edges))}
    
venny4py(sets=sets)

# %% computing edge labels
label_frequency = {}
tau_edges = G_gpt35.edges(data=True)
for edge in tau_edges:
    #print(edge)
    edge_type = edge[2]["type"]
    if edge_type in label_frequency:
        label_frequency[edge_type] = label_frequency[edge_type] + 1
    else:
        label_frequency[edge_type] = 1

print(label_frequency)
#label_frequency.pop("List") #only use for tau KG
D = sorted(label_frequency.items(),key=lambda x:x[1], reverse = True)
D_sorted_val = []
D_sorted_key = []
for sorted_item in D:
    D_sorted_key.append(sorted_item [0])
    D_sorted_val.append(sorted_item [1])

#D.pop("List") #only use for tau KG
plt.bar(range(len(D)), D_sorted_val, align='center')
plt.xticks(range(len(D)), D_sorted_key)
plt.xticks(rotation=90)
plt.title("GPT-3.5 KG")
plt.savefig('figure-paper/abstracts/gpt35-edge.png',bbox_inches='tight', dpi=200)
plt.show()

