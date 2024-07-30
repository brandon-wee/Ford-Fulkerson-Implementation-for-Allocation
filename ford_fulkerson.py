class FlowVertex:
    def __init__(self, vertex_id, index):
        """
        Constructor method for Vertex class, used for both Flow and Residual Networks
        Written by Brandon Wee Yong Jing

        Input:
            vertex_id: Integer representing the ID of the vertex
            index: Integer representing the Index of the vertex in the Flow/Residual Network
        Return:
            None

        time complexity: Best case and worst case is O(1).
        space complexity: Input: O(1), Auxiliary: O(1).
        """
        self.vertex_id = vertex_id
        self.sink = False
        self.source = False
        self.edges = []
        self.index = index
    #
    # def __repr__(self):
    #     return str(self.vertex_id) + ' ' + str(self.edges)


class FlowEdge:
    def __init__(self, start_vertex, end_vertex, capacity):
        """
        Constructor method for FlowEdge class, used for Flow Network
        Written by Brandon Wee Yong Jing

        Input:
            start_vertex: FlowVertex representing the start of the edge
            end_vertex: FlowVertex representing the end of the edge
            capacity: integer representing the capacity of the edge
        Return:
            None

        time complexity: Best case and worst case is O(1).
        space complexity: Input: O(1), Auxiliary: O(1).
        """
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.capacity = capacity
        self.flow = 0
        self.forward_residual_edge = None
        self.backward_residual_edge = None
    #
    # def __repr__(self):
    #     return f"{self.start_vertex.vertex_id} -- {self.capacity} -> {self.end_vertex.vertex_id}"


class FlowNetwork:
    def __init__(self, n):
        """
        Constructor method for Flow Network class.
        Written by Brandon Wee Yong Jing

        Input:
            n: Integer representing the number of vertices in the graph
        Return:
            None

        time complexity: Best case and worst case is O(n).
        space complexity: Input: O(1), Auxiliary: O(n).
        """
        self.graph = [FlowVertex(i, i) for i in range(n)]
        self.n = n
        self.source = None
        self.sink = None

    def add_edge(self, start, end, capacity):
        """
        Method used to add an edge between start and end vertices
        Written by Brandon Wee Yong Jing

        Input:
            start: Integer representing the index of the start vertex in the edge
            end: Integer representing the index of the end vertex in the edge
            capacity: Integer representing the capacity of the edge
        Return:
            None

        time complexity: Best case and worst case is O(1).
        space complexity: Input: O(1), Auxiliary: O(1).
        """
        new_edge = FlowEdge(self.graph[start], self.graph[end], capacity)
        self.graph[start].edges.append(new_edge)

    def define_sink_source(self, source_index, sink_index):
        """
        Method used to set the sink and source vertices in the flow network
        Written by Brandon Wee Yong Jing

        Input:
            source_index: Integer representing the index of the source vertex
            sink_index: Integer representing the index of the sink vertex
        Return:
            None

        time complexity: Best case and worst case is O(n).
        space complexity: Input: O(1), Auxiliary: O(n).
        """
        self.graph[sink_index].sink = True
        self.graph[source_index].source = True

        self.sink = sink_index
        self.source = source_index


class ResidualEdge:
    def __init__(self, flow_edge, flow, forwards=True):
        """
        Constructor method for Flow Network class.
        Written by Brandon Wee Yong Jing

        Input:
            flow_edge: FlowEdge representing the edge that will be converted to the residual edge for the residual network
            flow: Integer representing the flow of the residual edge
            forwards: Optional boolean with default value True to represent whether the edge is a forwards or backwards edge
        Return:
            None

        time complexity: Best case and worst case is O(1).
        space complexity: Input: O(1), Auxiliary: O(1).
        """
        self.flow_edge = flow_edge
        self.flow = flow
        self.forwards = forwards

        if forwards:
            self.start_vertex = flow_edge.start_vertex
            self.end_vertex = flow_edge.end_vertex
        else:
            self.start_vertex = flow_edge.end_vertex
            self.end_vertex = flow_edge.start_vertex

    # def __repr__(self):
    #     return f"{self.start_vertex.vertex_id} -- {self.flow} -> {self.end_vertex.vertex_id}"


class ResidualNetwork:
    def __init__(self, flow_network: FlowNetwork):
        """
        Constructor method for Residual Network class.
        Written by Brandon Wee Yong Jing

        Input:
            flow_network: FlowNetwork that will be converted into a residual network
        Return:
            None

        Let v be the number of vertices and e be the number of edges in the flow_network.
        time complexity: Best case and worst case is O(v + e).
        space complexity: Input: O(1), Auxiliary: O(v + e).
        """
        # Define residual network
        self.graph = [FlowVertex(i, i) for i in range(flow_network.n)]

        # Set source and sink of residual network
        self.source = flow_network.source
        self.sink = flow_network.sink
        self.graph[self.source].source = True
        self.graph[self.sink].sink = True
        self.residual_capacity = float('inf')

        # Add all edges to residual network
        for flow_vertex in flow_network.graph:
            for flow_edge in flow_vertex.edges:
                forward_edge = ResidualEdge(flow_edge, flow_edge.capacity)
                backward_edge = ResidualEdge(flow_edge, 0, False)

                start = flow_edge.start_vertex
                end = flow_edge.end_vertex

                for residual_edge in self.graph[start.vertex_id].edges:
                    # Update residual edge flow if edge already exist instead of creating a new edge
                    if residual_edge.start_vertex.vertex_id == start.vertex_id and residual_edge.end_vertex.vertex_id == end.vertex_id:
                        residual_edge.flow += forward_edge.flow
                        forward_edge = residual_edge
                        break
                else:
                    # Create new forward residual edge
                    self.graph[start.vertex_id].edges.append(forward_edge)

                for residual_edge in self.graph[end.vertex_id].edges:
                    # Update residual edge flow if edge already exist instead of creating a new edge
                    if residual_edge.start_vertex.vertex_id == end.vertex_id and residual_edge.end_vertex.vertex_id == start.vertex_id:
                        residual_edge.flow += backward_edge.flow
                        backward_edge = residual_edge
                        break
                else:
                    # Create new backward residual edge
                    self.graph[end.vertex_id].edges.append(backward_edge)

                flow_edge.forward_residual_edge = forward_edge
                flow_edge.backward_residual_edge = backward_edge

    def get_hasAugmentingPath(self):
        """
        Get an augmenting path in residual network using BFS.
        Written by Brandon Wee Yong Jing

        Input:
            None
        Return:
            path: A list representing the augmenting path

        Let v be the number of vertices and e be the number of edges in the flow_network.
        time complexity: Best case and worst case is O(v + e).
        space complexity: Input: O(1), Auxiliary: O(v + e).
        """
        visited = [False] * len(self.graph)
        previous = [None] * len(self.graph)
        queue = deque([self.graph[self.source]])

        # BFS to traverse residual network in order to obtain augmenting path
        while queue:
            current = queue.popleft()
            if visited[current.index]:
                continue

            visited[current.index] = True
            # If augmenting path is found
            if current.sink:
                # Backtrack to obtain the augmenting path
                return self.backtrack(previous, current.index)

            for neighbours in current.edges:
                if not visited[neighbours.end_vertex.index]:
                    if neighbours.flow != 0:
                        previous[neighbours.end_vertex.index] = neighbours
                        queue.append(self.graph[neighbours.end_vertex.index])

        # No augmenting path found, so return empty list
        return []

    def backtrack(self, previous, end_idx):
        """
        Backtrack method used to obtain the augmenting path from the end vertex. It also sets the residual capacity as an
        instance variable.
        Written by Brandon Wee Yong Jing

        Input:
            previous: list of visited vertices
            end_idx: integer representing the end vertex
        Return:
            path: list of edges that forms the augmenting path

        Let v be the number of vertices.
        time complexity: Best case and worst case is O(v).
        space complexity: Input: O(v), Auxiliary: O(v).
        """
        path = []
        residual_capacity = float('inf')

        # Backtrack to obtain the path
        while previous[end_idx] is not None:
            path.append(previous[end_idx])
            end_idx = previous[end_idx].start_vertex.index

            # Find the residual capacity
            residual_capacity = min(residual_capacity, abs(path[-1].flow))

        # Set residual capacity
        self.residual_capacity = residual_capacity
        return path[::-1]

    def augmentFlow(self, path):
        """
        Method to augment the flow in the residual network
        Written by Brandon Wee Yong Jing

        Input:
            path: list of edges that forms the augmenting path
        Return:
            None

        Let v be the number of vertices.
        time complexity: Best case and worst case is O(v).
        space complexity: Input: O(v), Auxiliary: O(v).
        """

        # Augment every edge in residual network
        for residual_edge in path:
            flow_edge = residual_edge.flow_edge
            flow_edge.flow += self.residual_capacity

            forward_edge = flow_edge.forward_residual_edge
            backward_edge = flow_edge.backward_residual_edge

            forward_edge.flow -= self.residual_capacity
            backward_edge.flow += self.residual_capacity


def FordFulkerson(graph):
    """
    Standard implementation of ford fulkerson
    Written by Brandon Wee Yong Jing

    Input:
        path: list of edges that forms the augmenting path
    Return:
        None

    Let v be the number of vertices and e be the number of edges in the flow network.
    time complexity: Best case and worst case is O(ve^2).
    space complexity: Input: O(v + e), Auxiliary: O(v + e).
    """
    flow = 0
    residual_network = ResidualNetwork(graph)

    # Continues to augment path until an augmenting path cannot be formed
    while path := residual_network.get_hasAugmentingPath():
        flow += residual_network.residual_capacity
        residual_network.augmentFlow(path)

    return flow