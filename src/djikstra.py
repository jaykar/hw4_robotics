def djikstra(nodes, S, G):
    #nodes = graph.nodes
    unvisited = set(nodes.keys())
    unvisited.remove(S)

    # initialize distance dictionary    
    dist = {}
    predecessors = {}
    dist[S] = 0;
    predecessors[S] = S
    S_neighbors = set(nodes[S].keys());
    for n in unvisited:
        if n in S_neighbors:
            dist[n] = nodes[S][n]
            predecessors[n] = S
        else:
            dist[n] = float('Inf');
   
    while len(unvisited) > 0:
        # find the closest unvisited node        
        keys = unvisited.intersection(set(dist.keys()))
        dist_of_unvisited = {k:dist[k] for k in keys}
        V = min(dist_of_unvisited, key = dist_of_unvisited.get);
        unvisited.remove(V);
        V_neighbors = set(nodes[V].keys())
        for W in V_neighbors:
            if ((dist[V] + nodes[V][W]) < dist[W]):
                dist[W] = dist[V] + nodes[V][W]
                predecessors[W] = V           
    path = []
    end = G;
    while end != S:
        path.append(end)
        end = predecessors[end]
    path.reverse()
    return path

nodes = {'G': {'F': 1, 'D': 1}, 'B': {'C': 2, 'F': 3}, 'F': {'G': 1, 'E': 2, 'B': 3}, 'D': {'G': 1, 'C': 1, 'A': 2}, 'E': {'C': 3, 'F': 2}, 'A': {'C': 1, 'D': 2}, 'C': {'E': 3, 'B': 2, 'A': 1, 'D': 1}}
path = djikstra(nodes, 'A', 'F')
print(path)