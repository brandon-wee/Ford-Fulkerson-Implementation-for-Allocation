
from ford_fulkerson import FlowNetwork, FordFulkerson


def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
    Returns the allocate list containing the allocations

    Approach:
    Let n be the number of security officers and m be the number of companies.
    To allocate based on the preferences, officers_per_org, min_shifts and max_shifts, we will need to form a flow
    network based on the given input.

    The flow_graph[0] is the source vertex that will connect to the original source and all of the
    security officer vertices.

    The flow_graph[1] is the original source vertex (in the circulation network) and will connect to all of the
    security officer vertices.

    The flow_graph[2], flow_graph[3], ..., flow_graph[n + 1] is the security officer vertices, each of them will connect
    to their 30 day vertices.

    The flow_graph[n + 2], flow_graph[n + 3], ..., flow_graph[n + 1 + 30n) is the day vertices, each of them will connect
    to the company shifts for each day.

    The flow_graph[31n + 2], flow_graph[31n + 3], ..., flow_graph[31n + 1 + 90m] is the company shifts for each day vertex,
    each of them will connect to the super sink node.

    The flow_graph[-1] is the super sink node.

    We can then run ford fulkerson on this flow network, and use the flow network to form the allocation list.

    Written by Brandon Wee Yong Jing.

    Input:
        preference: list of lists representing the preferences of each security guard
        officers_per_org: list of lists representing the number of officers each company will require for each shift
        min_shifts: integer representing the minimum number of shifts each officer is expected to take per month
        max_shifts: Integer representing the maximum number of shifts each officer can take per month
    Return:
        allocation: 4-D list that represents the allocation based on parameters.

    Time complexity:

        Best case analysis: O(n*n*m)
        Constructing the flow network takes O(m) vertices and O(n) edges. Hence, the complexity of running ford fulkerson
        is O(m*n^2) and constructing the allocation list takes O(m*n^2).

        Worst case analysis O(n*n*m)
        Same justification as best case analysis

    Space complexity:
        Input space analysis: O(n + m)
        preferences has space complexity O(n) and officers_per_org has space complexity O(m). Hence, the sum is O(n + m)

        Aux space analysis: O(n * n * m)
        Allocation list takes up O(m * n * n) space complexity.
    """

    # Obtain the number of officers, number of companies, number of days
    number_of_officers = len(preferences)
    number_of_companies = len(officers_per_org)
    days = 30

    # Sum all of the allocation required from the companies per day
    sum_of_allocation = 0
    for i in officers_per_org:
        for j in i:
            sum_of_allocation += j

    # Only consider generating flow network graph if the number
    if number_of_officers * min_shifts - sum_of_allocation * days <= 0:

        # Create flow network graph
        flow_graph = FlowNetwork(
            1 + 1 + number_of_officers + days * number_of_officers + 3 * days * number_of_companies + 1)

        # Connect Source to Original Source
        flow_graph.add_edge(0, 1, sum_of_allocation * days - number_of_officers * min_shifts)

        # Connect Source to Security Officers
        for i in range(2, number_of_officers + 2):
            flow_graph.add_edge(0, i, min_shifts)
            flow_graph.add_edge(1, i, max_shifts - min_shifts)

        # Connect Officers to Days
        for i in range(2, number_of_officers + 2):
            for j in range(1, days + 1):
                flow_graph.add_edge(i, i + j * number_of_officers, 1)

        # Connect Days to Company Shift Days
        for i in range(number_of_officers):
            s0, s1, s2 = map(int, preferences[i])
            temp = 0
            for j in range(number_of_officers + 2 + i, (days + 1) * number_of_officers + 2, number_of_officers):
                day_index = (days + 1) * number_of_officers + 2 + 3 * temp * number_of_companies
                shift = 0
                for k in range(day_index, day_index + 3 * number_of_companies):
                    if shift == 0:
                        flow_graph.add_edge(j, k, s0)

                    if shift == 1:
                        flow_graph.add_edge(j, k, s1)

                    if shift == 2:
                        flow_graph.add_edge(j, k, s2)

                    shift = (shift + 1) % 3

                temp += 1

        # Connect Company Shift Days to Sink
        for i in range(number_of_companies):
            company_day_index = (days + 1) * number_of_officers + 2

            for j in range(company_day_index + 3 * i, company_day_index + 3 * days * number_of_companies + 3 * i,
                           3 * number_of_companies):
                for k in range(3):
                    flow_graph.add_edge(j + k, -1, officers_per_org[i][k])

        # Set the sink and source vertices
        flow_graph.define_sink_source(0, -1)

        # Run ford fulkerson
        flow = FordFulkerson(flow_graph)

        # Flow network is valid if the flow of the graph is equal to the sum of allocations for day * 30 days
        if flow == days * sum_of_allocation:
            ans = [[[[0 for _ in range(3)] for _ in range(days)] for _ in range(number_of_companies)] for _ in
                   range(number_of_officers)]

            # Fill in allocation based on flow graph
            for i in range(number_of_officers):  # For every officer
                for officer_day_edge in flow_graph.graph[i + 2].edges:  # Iterate through every edge
                    if officer_day_edge.flow == 1:  # IF edge has flow of 1
                        # calculate day
                        d = (
                                        officer_day_edge.end_vertex.vertex_id - officer_day_edge.start_vertex.vertex_id) // number_of_officers - 1
                        for day_company_edge in officer_day_edge.end_vertex.edges:  # Iterate through every company edge
                            if day_company_edge.flow == 1:  # If company edge has flow of 1
                                offset = 1 + 1 + number_of_officers + days * number_of_officers  # Calculate offset

                                # Calculate k
                                k = (day_company_edge.end_vertex.vertex_id - 3 * d * number_of_companies - offset) % 3

                                # Calculate j
                                j = (day_company_edge.end_vertex.vertex_id - 3 * d * number_of_companies - offset) // 3

                                # Set allocation
                                ans[i][j][d][k] = 1
            return ans
        else:
            return None