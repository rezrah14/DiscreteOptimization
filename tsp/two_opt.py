import math
from collections import namedtuple
from typing import List

from tsp.plot import RoutePlot

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


def calculate_route_cost(cost_mat, route):
    # TODO: old_cost - (lengths of the removed edges) + (lengths of added edges)
    total_dist = 0.0
    for i in range(len(route) - 1):
        curr_idx = route[i]
        next_idx = route[i+1]

        dist = cost_mat[curr_idx][next_idx]
        total_dist += dist
    return total_dist


def two_opt(cost_matrix, init_toure, improvement_threshold=0.001):
    # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    # Initialize the improvement factor.
    improvement_factor = 1
    best_distance = calculate_route_cost(cost_matrix, init_toure)
    route = init_toure
    best = init_toure
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue  # changes nothing, skip then
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2woptSwap
                if calculate_route_cost(cost_matrix, new_route) < calculate_route_cost(cost_matrix, best):
                    best = new_route
                    improved = True
        route = best
    return best, None


def solve_it(input_data):
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    cost_matrix = create_cost_matrix(points)
    init_toure = get_initial_tour(points, cost_matrix)

    two_opt_plot = RoutePlot(points, init_toure)

    solution, cost_list = two_opt(cost_matrix, init_toure)

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

with open("input_file", 'r') as file:
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