import math
import random
from collections import namedtuple
from typing import List
import random

from plot import RoutePlot, CostPlot

Point = namedtuple("Point", ['x', 'y'])


def length(point1: Point, point2: Point):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def create_cost_matrix(points: List[Point]) -> List[List[float]]:
    node_count = len(points)
    matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
    for i in range(node_count):
        for j in range(i, node_count):
            matrix[i][j] = length(points[i], points[j])
            matrix[j][i] = matrix[i][j]
    return matrix


def get_initial_tour(points, cost_matrix, init_idx=0):
    idx = init_idx
    tour = []
    visited = set()

    while True:
        visited.add(idx)
        tour.append(idx)
        if len(visited) >= len(points):
            break
        dists = cost_matrix[idx]
        # TODO: dist_to_idx_dic -> sort based on smallest dist -> get the first idx not visited
        best_dist = float('inf')
        for i in range(len(points)):
            if i in visited:
                continue
            dist = dists[i]
            if dist < best_dist:
                best_dist = dist
                idx = i
    tour.append(tour[0])
    return tour


def get_initial_tours(points, cost_matrix, num_tours=10):
    tours = []
    num_points = len(points)
    num_tours = min(num_tours, num_points)
    visited = set()
    for _ in range(num_tours):
        init_idx = random.randint(0, num_points - 1)
        if init_idx in visited:
            continue
        visited.add(init_idx)
        tour = get_initial_tour(points, cost_matrix, init_idx)
        plot = RoutePlot(points)
        plot.add_route(tour)
        tours.append(tour)
    return tours


def calculate_route_cost(cost_mat, route):
    # TODO: old_cost - (lengths of the removed edges) + (lengths of added edges)
    total_dist = 0.0
    for i in range(len(route) - 1):
        curr_idx = route[i]
        next_idx = route[i+1]

        dist = cost_mat[curr_idx][next_idx]
        total_dist += dist
    return total_dist


def two_opt_tsp_simulated_annealing(cost_matrix, init_tours, route_plot=None):
    best_cost = float('inf')
    best_route = init_tours[0]

    while len(init_tours) > 0:
        rnd = random.randint(0, len(init_tours) - 1)
        route = init_tours.pop(rnd)
        num_nodes = len(route) - 1

        idx_to_neighbors = {}
        for n in range(num_nodes):
            neighbors = []
            for i in range(num_nodes):
                if 1 < math.fabs(n - i) < num_nodes - 1:
                    neighbors.append((i, cost_matrix[n][i]))
            neighbors.sort(key=lambda tup: tup[1])
            idx_to_neighbors[n] = neighbors

        route_cost = calculate_route_cost(cost_matrix, route)
        init_tmp = route_cost / num_nodes
        temp = init_tmp
        improved_cost = route_cost
        improved_route = route
        improved = True
        cost_list = []
        not_improve_counter = 0
        while temp > init_tmp * 0.001 and not_improve_counter < 20:
            if improved:
                rnd = random.randint(1, 100) / 1000
                coeff = 1.0 + rnd
                temp *= coeff
                not_improve_counter = 0
                update_search_neighborhood(cost_matrix, idx_to_neighbors, num_nodes)
            else:
                rnd = random.randint(1, 100) / 1000
                coeff = 1.0 + rnd
                temp /= coeff
                not_improve_counter += 1
            print(temp)

            rnd = random.uniform(0, 1)
            if rnd > 0.5:
                route.reverse()

            improved = False
            not_visited = [ii for ii in range(num_nodes)]
            while len(not_visited) > 0:
                idx = random.randint(0, len(not_visited) - 1)
                idx = not_visited[idx]

                not_visited.remove(idx)
                not_visited_neighbors = idx_to_neighbors[idx][:]
                num_not_visited_neighbors = len(not_visited_neighbors) - 1
                for _ in range(num_not_visited_neighbors):
                    i = random.randint(0, len(not_visited_neighbors) - 1)
                    tup = not_visited_neighbors[i]
                    i = tup[0]
                    not_visited_neighbors.remove(tup)

                    new_route = route[0:-1]
                    min_idx = min(idx, i)
                    max_idx = max(idx, i)

                    for j in range(max_idx - min_idx + 1):
                        new_route[min_idx + j] = route[max_idx - j]
                    new_route.append(new_route[0])

                    cost = calculate_route_cost(cost_matrix, new_route)
                    cost_list.append(cost)

                    if cost < improved_cost:
                        improved_route = new_route
                        improved_cost = cost
                        if route_plot is not None:
                            route_plot.update_data(improved_route, improved_cost, improved_cost)
                        improved = True
                        route = new_route
                        update_search_neighborhood(cost_matrix, idx_to_neighbors, num_nodes)
                    else:
                        delta = cost - improved_cost
                        prob = math.exp(-delta / temp)
                        rnd = random.uniform(0, 1)
                        if prob >= rnd:
                            if route_plot is not None:
                                route_plot.update_data(new_route, cost, improved_cost)
                            route = new_route

            route = improved_route

        if improved_cost < best_cost:
            best_cost = improved_cost
            best_route = improved_route
            init_tours.append(improved_route)
        else:
            rnd = random.uniform(0, 1)
            if rnd > 0.8:
                init_tours.append(improved_route)

    route_plot.update_data(best_route, best_cost, best_cost)
    return best_route, cost_list


def update_search_neighborhood(cost_matrix, idx_to_neighbors, num_nodes):
    for n in range(num_nodes):
        neighbors = idx_to_neighbors[n]
        if n == 0:
            edge_len_in = cost_matrix[n][num_nodes - 1]
        else:
            edge_len_in = cost_matrix[n][n - 1]

        if n == num_nodes - 1:
            edge_len_out = cost_matrix[n][0]
        else:
            edge_len_out = cost_matrix[n][n + 1]

        max_edge_len = max(edge_len_in, edge_len_out) * 2.5
        trimmed_neighbors = []
        for ind, dis in neighbors:
            if dis > max_edge_len and len(trimmed_neighbors) > 0.1 * num_nodes:
                break
            trimmed_neighbors.append((ind, dis))
        idx_to_neighbors[n] = trimmed_neighbors


def solve_it(input_data):
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    cost_matrix = create_cost_matrix(points)
    init_tours = get_initial_tours(points, cost_matrix, math.ceil(len(points) * .1))
    init_cost = calculate_route_cost(cost_matrix, init_tours[0])

    two_opt_plot = RoutePlot(points)
    two_opt_plot.add_route(init_tours[0], init_cost)

    solution, cost_list = two_opt_tsp_simulated_annealing(cost_matrix, init_tours, two_opt_plot)

    CostPlot(cost_list)
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    pass
    # if len(sys.argv) > 1:
    #     file_location = sys.argv[1].strip()
    #     with open(file_location, 'r') as input_data_file:
    #         input_data = input_data_file.read()
    #     print(solve_it(input_data))
    # else:
    #     print('This test requires an input file.  Please select one from the data directory. '
    #           '(i.e. python solver.py ./data/tsp_51_1)')

with open("input_data_file", 'r') as file:
    input_list = file.read().split()
    if len(input_list) > 1:
        for file_location in input_list:
            with open("data/" + file_location, 'r') as input_data_file:
                input_data = input_data_file.read()
                sln = solve_it(input_data)
                with open(f"res_{file_location}", "w") as output_file:
                    output_file.write(sln)
                print(sln)
    else:
        print('This test requires an input file.  Please select one from the data directory.'
              ' (i.e. python solver.py ./data/ks_4_0)')