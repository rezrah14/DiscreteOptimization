import math
import random
from collections import namedtuple
from typing import List
import random

from tsp.plot import RoutePlot, CostPlot

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


def get_initial_tour(points, cost_matrix):
    idx = 0
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


def get_initial_tours(points, cost_matrix):
    tours = []
    num_points = len(points)
    num_init_tours = math.ceil(0.10 * num_points)

    for _ in range(num_init_tours):
        tour = []
        idx = random.randint(0, num_points - 1)
        tour.append(idx)
        visited = set()

        while len(tour) < num_points:
            visited.add(idx)
            nei_dists = cost_matrix[idx][:]
            dist_to_nei_idx = {dist: j for j, dist in enumerate(nei_dists)}
            nei_dists = [d for d in nei_dists if dist_to_nei_idx[d] not in visited]
            nei_dists.sort()
            nei_idx = random.randint(0, min(len(nei_dists), num_init_tours) - 1)
            nei_idx = nei_dists[nei_idx]
            nei_idx = dist_to_nei_idx[nei_idx]
            #del dist_to_nei_idx[nei_idx]
            tour.append(nei_idx)
            idx = nei_idx

        tour.append(tour[0])
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


def two_opt(cost_matrix, init_toure, route_plot=None, improvement_threshold=0.001):
    # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    best_distance = calculate_route_cost(cost_matrix, init_toure)
    route = init_toure
    best = init_toure
    improved = True
    distances = []
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2woptSwap
                dist = calculate_route_cost(cost_matrix, new_route)
                distances.append(dist)
                if dist < best_distance:
                    best = new_route
                    best_distance = dist
                    if route_plot is not None:
                        route_plot.update_data(best, None)
                    improved = True

        route = best
    return best, distances


class Node:
    def __init__(self, index: int, sz: int, prv: int, nxt: int):
        self.index = index
        self.prev = prv
        self.next = nxt
        self.neighbors = [i for i in range(sz - 1) if (i != index and i != prv and i != nxt)]


def two_opt_ii(cost_matrix, init_route, route_plot=None, improvement_threshold=0.001):
    best_distance = calculate_route_cost(cost_matrix, init_route)
    best_route = init_route
    improved = True
    distances = []
    route = init_route

    for _ in range(50):
        while improved:
            improved = False
            not_visited = [i for i in range(len(route) - 1)]
            while len(not_visited) > 0:
                idx = random.randint(0, len(not_visited) - 1)
                idx = not_visited[idx]
                # idx = 0
                not_visited.remove(idx)
                for i in range(len(route) - 1):
                    if (idx + 1) > len(route) - 1:
                        idx = 0

                    if math.fabs(i - idx) <= 1:
                        continue

                    new_route = route[0:-1]
                    min_idx = min(idx, i)
                    max_idx = max(idx, i)

                    for j in range(max_idx - min_idx + 1):
                        new_route[min_idx + j] = route[max_idx - j]
                    new_route.append(new_route[0])

                    dist = calculate_route_cost(cost_matrix, new_route)
                    distances.append(dist)

                    if dist < best_distance:
                        best_route = new_route
                        best_distance = dist
                        if route_plot is not None:
                            route_plot.update_data(best_route, None)
                        improved = True
                        route = new_route

    return best_route, distances


def two_opt_iii(cost_matrix, init_routs, route_plot=None, improvement_threshold=0.001):
    best_distance = calculate_route_cost(cost_matrix, init_routs[0])
    best_route = init_routs[0]
    improved = True
    distances = []
    route = init_routs[0]
    counter = 0
    for init_route in init_routs:
        T = 10
        really_improved = False
        all_best_routes = [init_route]
        while T > 0.001:
            ii = random.randint(0, len(all_best_routes) - 1)
            route = all_best_routes[ii]
            improved = True
            if really_improved:
                T *= 1.11
            else:
                T *= 0.99
            print(T)
            really_improved = False
            while improved:
                improved = False
                not_visited = [i for i in range(len(route) - 1)]
                while len(not_visited) > 0:
                    idx = random.randint(0, len(not_visited) - 1)
                    idx = not_visited[idx]
                    counter = 0
                    not_visited.remove(idx)
                    not_visited_neighbors = [i for i in range(len(route) - 1)]
                    for _ in range(len(route) - 1):
                        if counter > int(len(route) * 0.2):
                            break
                        i = random.randint(0, len(not_visited_neighbors) - 1)
                        i = not_visited_neighbors[i]
                        not_visited_neighbors.remove(i)
                        if (idx + 1) > len(route) - 1:
                            idx = 0

                        if math.fabs(i - idx) <= 1:
                            continue

                        new_route = route[0:-1]
                        min_idx = min(idx, i)
                        max_idx = max(idx, i)

                        for j in range(max_idx - min_idx + 1):
                            new_route[min_idx + j] = route[max_idx - j]
                        new_route.append(new_route[0])

                        dist = calculate_route_cost(cost_matrix, new_route)
                        distances.append(dist)

                        if dist < best_distance:
                            best_route = new_route
                            old_best_dist = best_distance
                            best_distance = dist
                            if route_plot is not None:
                                route_plot.update_data(best_route, best_distance, best_distance)
                            improved = True
                            route = new_route
                            counter = 0
                            really_improved = True
                            if (old_best_dist - best_distance) / old_best_dist >= 0.005:
                                all_best_routes.append(best_route)
                            # T *= 1.1
                            break
                        else:
                            delta = dist - best_distance
                            prob = math.exp(-delta / T)
                            rnd = random.uniform(0, 1)
                            if prob > rnd:
                                improved = True
                                if route_plot is not None:
                                    route_plot.update_data(new_route, dist, best_distance)
                                route = new_route
                                break
                            else:
                                counter += 1
                                # T *= 0.9
                                # break

    return best_route, distances


def solve_it(input_data):
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    cost_matrix = create_cost_matrix(points)
    init_tours = get_initial_tours(points, cost_matrix)
    #init_toure = [0, 1, 2, 3, 5, 4, 0]
    init_cost = calculate_route_cost(cost_matrix, init_tours[0])

    two_opt_plot = RoutePlot(points)
    two_opt_plot.add_route(init_tours[0], init_cost)

    solution, cost_list = two_opt_iii(cost_matrix, init_tours, two_opt_plot)

    bench = [0, 5, 2, 28, 10, 9, 45, 3, 27, 41, 24, 46, 8, 4, 34, 23, 35, 13, 7, 19,
             40, 18, 16, 44, 14, 15, 38, 50, 39, 49, 17, 32, 48, 22, 31, 1, 25, 20,
             37, 21, 43, 29, 42, 11, 30, 12, 36, 6, 26, 47, 33, 0]
    cost = calculate_route_cost(cost_matrix, bench)
    two_opt_plot.add_route(bench, cost, color='blue')
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