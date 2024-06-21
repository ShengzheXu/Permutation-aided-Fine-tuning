import os
import parse
import pandas as pd
import ast
import networkx as nx

def parse_dictionary_hyfd(df_name):
    dir = 'fd_finder/'
    df = pd.read_csv(f"{dir}datasets/{df_name}.csv")
    cols = list(df.columns)
    report = open(f"{dir}json/{df_name}.json", "r").read()
    results = ast.literal_eval(report)

    cnt = 0
    fd_results = {key: [] for key in df.columns}
    for result in results:
        lhs_full = []
        for lhs in result[0]:
            lhs_full.append(cols[lhs])
        if len(lhs_full)==1: lhs_full = lhs_full[0]
        rhs_full = []
        for rhs in result[1]:
            rhs_full.append(cols[rhs])
        cnt = cnt + len(rhs_full)
        
        if str(lhs_full) not in fd_results.keys():
            fd_results[str(lhs_full)] = rhs_full
        else:
            fd_results[str(lhs_full)] = rhs_full
    
    return fd_results

# Using HyFD
# python3 hyfd/hyfd.py datasets/us_location.csv


if __name__ == "__main__":
    fd_results_pyro = {}
    # load FD results from HyFD
    files = os.listdir("./fd_finder/datasets")
    
    sorted_order = {}
    for f in files:
        name = f.split('/')[-1].split('.')[0]
        print('='*15, name)

        fd_list = parse_dictionary_hyfd(name)
        print(fd_list)

        G = nx.DiGraph()

        for key in fd_list:
            if '[' in key:
                kk = key[1:-1].split(', ')
                kk = [k[1:-1] for k in kk]
                for value in fd_list[key]:
                    for k in kk:
                        G.add_edge(value, k)
            else:
                for value in fd_list[key]:
                    G.add_edge(key, value)

        # if G is empty
        if len(G.nodes) == 0:
            print("The graph is empty.")
            sorted_order[name] = ','.join(fd_list.keys())
            print(sorted_order[name])
            continue

        if nx.is_directed_acyclic_graph(G):
            print("The graph is already a DAG.")
            print('Recommended order:')
            sorted_nodes = list(nx.topological_sort(G))
            sorted_order[name] = ','.join(sorted_nodes)
            print(sorted_nodes)
        else:
            sccs = list(nx.strongly_connected_components(G))
            DAG_without_cycles = nx.condensation(G, scc=sccs)
            sorted_nodes = list(nx.topological_sort(DAG_without_cycles))
            sorted_nodes_full = []
            sorted_nodes_just_list = []
            for node in sorted_nodes:
                sorted_nodes_full.append(list(DAG_without_cycles.nodes[node]['members']))
                sorted_nodes_just_list.extend(list(DAG_without_cycles.nodes[node]['members']))
            print('Recommended order:')
            print(sorted_nodes_full)
            sorted_order[name] = ','.join(sorted_nodes_just_list)

    with open('feature_order_permutation.txt', 'w') as file:
        file.write(str(sorted_order))


