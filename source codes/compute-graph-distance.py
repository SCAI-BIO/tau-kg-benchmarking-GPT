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
# import networkx as nx
# G_gpt4 = nx.Graph(driver)   # undirected graph
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
# query = """
# # MATCH (n)-[r]->(c) RETURN *
# # """
# results = driver.session().run(query)
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
# query = """
# # MATCH (n)-[r]->(c) RETURN *
# # """
# results = driver.session().run(query)
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
# query = """
# # MATCH (n)-[r]->(c) RETURN *
# # """
# results = driver.session().run(query)
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

# %%
# from ged4py.algorithm import graph_edit_dist
# print(graph_edit_dist.compare(G_tau,G_sherpa))

response = nx.optimize_edit_paths(G_gpt4, G_sherpa)
print("done optimize path computation")
for item in response:
    print(item)
print("done")
#%% intersection of nodes and edges uing library
inter_gpt4_gpt35 = nx.intersection(G_gpt4, G_gpt35)
set_inter_gpt4_gpt35 = inter_gpt4_gpt35.nodes()

inter_gpt4_sherpa = nx.intersection(G_gpt4, G_sherpa)
set_inter_gpt4_sherpa = inter_gpt4_sherpa.nodes()

inter_gpt4_tau = nx.intersection(G_gpt4, G_tau)
set_inter_gpt4_tau = inter_gpt4_tau.nodes()

inter_gpt35_sherpa = nx.intersection(G_gpt35, G_sherpa)
set_inter_gpt35_sherpa = inter_gpt35_sherpa.nodes()

inter_gpt35_tau = nx.intersection(G_gpt35, G_tau)
set_inter_gpt35_tau = inter_gpt35_tau.nodes()

inter_sherpa_tau = nx.intersection(G_sherpa, G_tau)
set_inter_sherpa_tau = inter_sherpa_tau.nodes()
print(inter_gpt35_tau.edges)
#%%
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
# print("edges")
set_tau_edges = G_tau.edges()
set_sherpa_edges = G_sherpa.edges()
set_gpt4_edges = G_gpt4.edges()
set_gpt35_edges = G_gpt35.edges()
# print("sherpa",len(set_sherpa))
# print("tau",len(set_tau))
# print("gpt4",len(set_gpt4))
# print("gpt35",len(set_gpt35))
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
#%%
for triple in B:
    edge_1 = triple[0]
    edge_2 = triple[1]
    query_query = """
    MATCH (p)-[r]->(m) where ID(p) = {} AND ID(m) = {} return p,r,m
    """.format(edge_1, edge_2)
    results_query= driver.session().run(query_query)
    G_query = nx.MultiGraph()
    nodes_query = list(results_query.graph()._nodes.values())
    for node in nodes_query:
        G_query.add_node(node.id, labels=node._labels, properties=node._properties)
    rels_query = list(results.graph()._relationships.values())
    for rel in rels_query:
        G_query.add_edge(rel.start_node.id, rel.end_node.id, key=rel.id, type=rel.type, properties=rel._properties)
#%% extract nodes that are just in one (Tau) graph, their most frequeny of types

just_in_one_set = set(set_tau) - set(set_tau).intersection(set(set_sherpa)).union(set(set_tau).intersection(set(set_gpt4()), set(set_tau).intersection(set(set_gpt35)))) 
print(len(just_in_one_set))
#%% draw figure venn 3d
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
venn2([set_tau, set_sherpa],set_labels = ('tau nodes', 'sherpa nodes'), set_colors=("orange", "blue"),alpha=0.7)
plt.savefig('figure-paper/nodes_tau-sherpa.png')
plt.show()

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
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
venn2([set_tau, set_gpt4],set_labels = ('tau nodes', 'gpt4 nodes'), set_colors=("orange", "blue"),alpha=0.7)
plt.savefig('figure-paper/nodes_tau-gpt4.png')
plt.show()

plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
venn2([set_tau, set_gpt35],set_labels = ('tau nodes', 'gpt35 nodes'), set_colors=("orange", "blue"),alpha=0.7)
plt.savefig('figure-paper/nodes_tau-gpt35.png')
plt.show()

#%%
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of nodes")
s = G_sherpa.nodes()
t = G_gpt4.nodes()
u= G_gpt35.nodes()
venn3([s,t,u],("sherpa","gpt4","gpt35"),set_colors=("orange", "blue","red"),alpha=0.7)
plt.savefig('figure-paper/nodes-three-tools.png')
plt.show()
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

#%%
# res = [ele1 for ele1 in gpt4_edges 
#        for ele2 in tau_edges if ele1 == ele2]
# print(len(res))

# %%
#plt.style.use('seaborn')
#plt.figure(figsize=(10,10))
#plt.title("Venn Diagram For Cheetahs v/s Leopards")

# %%
# selected_edges = [(u,v) for u,v,e in G_total.edges(data=True) if e['annotationDatasource'] == 'gpt4']
# print(selected_edges)
# %%
edges = nx.get_edge_attributes(G_total, 'properties')
print(edges)


#%%ven 4d for common relations without caring about evidences
from venny4py.venny4py import *
sets = {
    'Tau KG': set(list(tau_edges)),
    'GPT3.5': set(list(gpt35_edges)),
    'GPT4': set(list(gpt4_edges)),
    'Sherpa': set(list(sherpa_edges))}
    
venny4py(sets=sets)

#%%
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
s = set(tau_edges)
t = set(sherpa_edges)
venn2([s,t],("tau","sherpa"),set_colors=("orange", "blue"),alpha=0.7)
plt.savefig('figure-paper/tau-sherpa.png')
plt.show()

plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
s = set(tau_edges)
t = set(gpt4_edges)

venn2([s,t],("tau","gpt4"),set_colors=("orange", "blue"),alpha=0.7)
plt.savefig('figure-paper/tau-gpt4.png')
plt.show()

plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
s = set(tau_edges)
t = set(gpt35_edges)
venn2([s,t],("tau","gpt35"),set_colors=("orange", "blue"),alpha=0.7)
plt.savefig('figure-paper/tau-gpt35.png')
plt.show()

#%% Print intersection triples
s = set(tau_edges)
t = set(gpt35_edges)
intersect = set.intersection(s,t)
for item in intersect:
    node_start = item[0]
    node_end = item[2]
    rel = item[1]
    tuple = (node_start,rel,node_end)
    bel_triple = (G_total.nodes[node_start]['properties']['bel'], relation_att[tuple]['type'], G_total.nodes[node_end]['properties']['bel'])
    print(bel_triple)

# %%
plt.style.use('seaborn')
plt.figure(figsize=(10,10))
plt.title("Intersection of KGs")
s = set(gpt4_edges)
t = set(gpt35_edges)
u=set(sherpa_edges)
venn3([s,t,u],("gpt4","gpt35","sherpa"),set_colors=("orange", "blue","red"),alpha=0.7)
plt.savefig('figure-paper/three-tools.png')
plt.show()


# %%
print(node_annot)
# %% Computing node labels and bar blot
label_frequency = {}
#for nodes change: G_tau.nodes(data=True)
tau_nodes = G_gpt35.nodes(data=True)
for node in tau_nodes:
    #print(node[2])
    label_set = node[1]["labels"]
    for item in label_set:
        if item in label_frequency:
            label_frequency[item] = label_frequency[item] + 1
        else:
            label_frequency[item] = 1

print(label_frequency)
#label_frequency.pop("List") #only use for tau KG
D = sorted(label_frequency.items(),key=lambda x:x[1], reverse = True)
D_sorted_val = []
D_sorted_key = []
for sorted_item in D:
    D_sorted_key.append(sorted_item [0])
    D_sorted_val.append(sorted_item [1])

plt.bar(range(len(D)), D_sorted_val, align='center')
plt.xticks(range(len(D)), D_sorted_key)
plt.xticks(rotation=90)
plt.title("GPT-3.5 KG")
plt.savefig('figure-paper/abstracts/gpt35.png',bbox_inches='tight', dpi=200)
plt.show()
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
# %% node degree pie chart

query = 'MATCH (p)-[r]->(m) where r.annotationDatasource = ["gpt3.5"] RETURN m.bel, count(*) as degree ORDER BY degree DESC LIMIT 7'
results = driver.session().run(query)
print(list(results.graph()._nodes.values()))

# %% bar charts of node degrees Tau

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
counts = [153, 127, 101, 75, 48,48, 42, 40, 37, 37 ]
bar_labels = ['path(MESH:Alzheimer’s disease)', 'p(HGNC:MAPT)', 'a(MESH:Protein Aggregates)', 'a(MESH: tau Proteins)', "bp(GO: lysophagy)", "a(GO:host cell Cajal body)", "bp(GO.symbiont-mediated activation of host autophagy)","deg(p(HGNC:MAPT)", "p(MESH:SH2 Domain-Containing Protein Tyrosine Phosphatases", 'p(NCIT:17q21 Microdeletion Syndrome,var("p.Pro301Leu"))']
#bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
ax.bar(bar_labels, counts)
ax.set_ylabel('node degree')
plt.xticks(rotation=90)
ax.set_title('Node degree in Tau KG')
plt.savefig("figure-paper/node-degree-tau.png",bbox_inches='tight')
plt.show()

# %% GPT    4
fig, ax = plt.subplots()
counts = [187, 23, 20, 20, 17, 14, 13, 13, 13, 9 ]
bar_labels = ['p(HGNC:MAPT)', "path(DOID:Alzheimer's disease)", "p(HGNC:DAPK1)", "p(HGNC:TFEB)", "p(HGNC:SYK)", "bp(GO:T cell aggregation)", "p(HGNC:TTBK2)", "p(HGNC:SIRT1)", "p(HGNC:DYRK1A)", "p(HGNC:FKBP5)"]
#bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
ax.bar(bar_labels, counts)
ax.set_ylabel('node degree')
plt.xticks(rotation=90)
ax.set_title('Node degree in GPT-4 KG')
plt.savefig("figure-paper/node-degree-gpt4.png",bbox_inches='tight')
plt.show()
# %% gpt35
fig, ax = plt.subplots()
counts = [114, 90, 42, 40, 32, 30, 29, 26, 25, 25 ]
bar_labels = ["a(HGNC:MAPT)", "a(MESH:tau Proteins)", "path(MESH:Alzheimer Disease)", "a(HGNC:pSyk)", "p(HGNC:MAPT)", "a(MESH:brain)", "a(HGNC:TTBK2)", "a(HGNC:TFEB)", "a(MESH:Alzheimer Disease)", "a(HGNC:IGLVIV-64)"]
#bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
ax.bar(bar_labels, counts)
ax.set_ylabel('node degree')
plt.xticks(rotation=90)
ax.set_title('Node degree in GPT-3.5 KG')
plt.savefig("figure-paper/node-degree-gpt35.png",bbox_inches='tight')
plt.show()
# %% sherpa
fig, ax = plt.subplots()
counts = [141, 56, 30, 30, 23, 22, 22, 21, 21, 21 ]
bar_labels = ["p(HGNC:MAPT)", "p(HGNC:MAPT, pmod(GO: protein phosphorylation))", "path(MESH:tauopathy)", "p(HGNC:DYRK1A)", "p(HGNC:MEF2D)", "p(HGNC:TTBK2)", "path(UNKNOWN:CTE)", "p(HGNC:HSP90AA1)", "p(HGNC:DAPK1)", "p(HGNC:TFEB)"]
#bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
ax.bar(bar_labels, counts)
ax.set_ylabel('node degree')
plt.xticks(rotation=90)
ax.set_title('Node degree in Sherpa KG')
plt.savefig("figure-paper/node-degree-sherpa.png",bbox_inches='tight')
plt.show()

# %% test function return nodes only in tau and see distribution
from functools import reduce
A = [set(set_tau), set(set_sherpa), set(set_gpt4), set(set_gpt35)]
B = set()
for i in range(1):
    U = reduce(set.union, A[:i]+A[(i+1):])
    B = B.union(set.difference(A[i], U))

print(len(B))
class_distribution = {}
# % get nodes for a special id
for id in B:
    query = """
    MATCH (p)-[r]-(m)  where ID(m) = {} return p
    """.format(id)
    results = driver.session().run(query)
    data = results.data()
    data_dict = data[0]
    class_ = data_dict["p"]["bel"].split("(")[0]
    if class_ in class_distribution:
        class_distribution[class_] = class_distribution[class_] + 1
    else:
        class_distribution[class_] = 1

#%% create bar chart
#class_distribution.pop("list")
D = sorted(class_distribution.items(),key=lambda x:x[1], reverse = True)
D_sorted_val = []
D_sorted_key = []
for sorted_item in D:
    D_sorted_key.append(sorted_item [0])
    D_sorted_val.append(sorted_item [1])

plt.bar(range(len(D)), D_sorted_val, align='center')
plt.xticks(range(len(D)), D_sorted_key)
plt.xticks(rotation=90)
plt.title("Class distribution of nodes only in Tau KG")
plt.savefig('figure-paper/abstracts/tau_class_distirbution.png',bbox_inches='tight', dpi=200)
plt.show()


# %% get total number of components
data = pd.read_csv("neo4j-analysis-graph-shape/components.csv")

# %%
print(len(set(data["Component"])))
# %%
