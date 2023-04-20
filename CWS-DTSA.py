import re
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvrp_io import read_TSPLIB_CVRP
from verypy.classic_heuristics.paessens_savings import paessens_savings_init


# Take a route of given length, divide it into subroute where each subroute is assigned to a vehicle
def routeToSubroute(routes, file):
    """
        Inputs: Sequence of customers that a route has
                loaded instance problem
        Outputs: Route that is divided into subroutes
                which is assigned to each vehicle
        """
    route = []
    sub_route = []
    vehicle_load = 0
    vehicle_capacity = file.capacity_constraint
    customer_demand = file.customer_demands
    for customer_id in routes:
        demand = customer_demand[customer_id]
        updated_vehicle_load = vehicle_load + demand
        if demand != 0:
            if updated_vehicle_load <= vehicle_capacity:
                sub_route.append(customer_id)
                vehicle_load = updated_vehicle_load
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
    if sub_route:
        route.append(sub_route)
    for subroute in route:
        subroute.append(0)
        subroute.insert(0, 0)
    return route


# Given a route, give its total cost
def getRouteCost(input_tree, distance_matrix, unit_cost=1):
    """
    Inputs:
        - Individual route
        - Demands
        - Unit cost for the route (can be petrol etc.)
    Outputs:
        - Total cost for the route taken by all the vehicles
    """
    total_cost = 0
    for sub_route in input_tree:
        # Initializing the subroute distance to 0
        sub_route_distance = 0
        # Initializing customer id for depot as 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Distance from the last customer id to next one in the given subroute
            distance = distance_matrix[last_customer_id][customer_id]
            sub_route_distance = sub_route_distance + distance
            # Update last_customer_id to the new one
            last_customer_id = customer_id
        # After adding distances in subroute, adding the route cost from last customer to depot that is 0
        sub_route_distance = sub_route_distance + distance_matrix[last_customer_id][0]
        # Cost for this particular subroute
        sub_route_transport_cost = unit_cost * sub_route_distance
        # Adding this to total cost
        total_cost = total_cost + sub_route_transport_cost
    return total_cost


# Initialize solutions including the output of Parallel Paessens Clarke-Wright Savings Algorithm
def initializeTrees(route, num_vertices, population_size):
    indices = [i for i in range(num_vertices)]
    indices.pop(0)
    trees = [route]
    for i in range(population_size - 1):
        tree = random.sample(indices, num_vertices - 1)
        trees.append(tree)
    return trees


# Swap transformation operator
def swap(input_tree):
    tree = [vertex for vertex in input_tree]
    index = random.sample(list(range(len(tree))), k=2)
    tree[index[0]], tree[index[1]] = tree[index[1]], tree[index[0]]
    return tree


# Shift transformation operator
def shift(input_tree):
    tree = [vertex for vertex in input_tree]
    index = random.sample(list(range(len(tree))), k=2)
    while max(index) - min(index) == 1:
        index = random.sample(list(range(len(tree))), k=2)
    left_index = min(index)
    right_index = max(index)
    temp = tree[left_index]
    for i in range(left_index, right_index):
        tree[i] = tree[i + 1]
    tree[right_index] = temp
    return tree


# Symmetry transformation operator
def symmetry(input_tree):
    tree = [vertex for vertex in input_tree]
    size = random.randint(2, len(tree) // 2 - 2)
    block1_start = random.choice([x for x in range(len(tree))])
    block2_start = random.choice([x for x in range(len(tree)) if x not in [y % len(tree) for y in
                                                                           range(block1_start - size + 1,
                                                                                 block1_start + size)]])
    new_tree = [vertex for vertex in input_tree]
    for i in range(size):
        new_tree[(block1_start + i) % len(tree)] = tree[(block2_start + size - 1 - i) % len(tree)]
        new_tree[(block2_start + i) % len(tree)] = tree[(block1_start + size - 1 - i) % len(tree)]
    return new_tree


# Implementation of the main algorithm
def DTSA(route, distance_matrix, num_vertices, population_size, search_tendency, file):
    max_FE = population_size * 5000
    FE = 0
    trees = initializeTrees(route, num_vertices, population_size)
    distances = [getRouteCost(routeToSubroute(tree, file), distance_matrix) for tree in trees]
    FE += population_size
    best_tree_index = distances.index(min(distances))
    while FE < max_FE:
        new_trees = []
        new_distances = []
        for i in range(population_size):
            current_tree = trees[i]
            best_tree = trees[best_tree_index]
            random_tree = trees[random.choice([x for x in range(len(trees)) if x != i and x != best_tree_index])]
            seeds = []
            if random.random() <= search_tendency:
                seeds.append(swap(best_tree))
                seeds.append(shift(best_tree))
                seeds.append(symmetry(best_tree))
                seeds.append(swap(random_tree))
                seeds.append(shift(random_tree))
                seeds.append(symmetry(random_tree))
            else:
                seeds.append(swap(current_tree))
                seeds.append(shift(current_tree))
                seeds.append(symmetry(current_tree))
                seeds.append(swap(random_tree))
                seeds.append(shift(random_tree))
                seeds.append(symmetry(random_tree))
            seed_distances = [getRouteCost(routeToSubroute(seed, file), distance_matrix) for seed in seeds]
            FE += 6
            best_seed_distance = min(seed_distances)
            best_seed = seeds[seed_distances.index(best_seed_distance)]
            if best_seed_distance < distances[i]:
                new_trees.append(best_seed)
                new_distances.append(best_seed_distance)
            else:
                new_trees.append(trees[i])
                new_distances.append(distances[i])
        trees = [tree for tree in new_trees]
        distances = [distance for distance in new_distances]
        best_tree_index = distances.index(min(distances))
    return trees, distances, best_tree_index


# 2-Opt heuristic functions
def route_distance(route, dist_matrix):
    dist = 0
    for i in range(len(route) - 1):
        dist += dist_matrix[route[i]][route[i + 1]]
    return dist + dist_matrix[route[-1]][route[0]]


def total_distance(subroutes, dist_matrix):
    return sum(route_distance(route, dist_matrix) for route in subroutes)


def is_feasible(subroute):
    return subroute[0] == subroute[-1] == 0


# 2-Opt swap between routes
def two_opt_swap(subroute1, subroute2):
    for i in range(len(subroute1)):
        for j in range(len(subroute2)):
            if i == 0 and j == 0:
                continue
            new_subroute1 = subroute1[:i] + subroute2[j:]
            new_subroute2 = subroute2[:j] + subroute1[i:]
            if is_feasible(new_subroute1) and is_feasible(new_subroute2):
                return new_subroute1, new_subroute2
    return subroute1, subroute2


# Main 2-Opt implementation
def two_opt(subroutes, dist_matrix):
    improved = True
    best_distance = total_distance(subroutes, dist_matrix)
    while improved:
        improved = False
        for i in range(len(subroutes)):
            for j in range(i + 1, len(subroutes)):
                new_subroutes = subroutes[:]
                new_subroutes[i], new_subroutes[j] = two_opt_swap(subroutes[i], subroutes[j])
                new_distance = total_distance(new_subroutes, dist_matrix)
                if new_distance < best_distance:
                    subroutes = new_subroutes
                    best_distance = new_distance
                    improved = True
    return subroutes


# Saves the visualization of the graph of the input subroutes
def plotSubroute(subroute, data, color):
    totalSubroute = [0] + subroute + [0]
    subroutelen = len(subroute)
    for i in range(subroutelen + 1):
        firstcust = totalSubroute[0]
        secondcust = totalSubroute[1]
        plt.plot([data.x[firstcust], data.x[secondcust]], [data.y[firstcust], data.y[secondcust]], c=color)
        totalSubroute.pop(0)


def plotRoute(trees, num_vertices, data,
              directory, filename
              ):
    colorslist = ["#de6262", "#dea062", "#c3de62", "#94de62", "#62dea2",
                  "#62dade", "#627fde", "#a862de", "#d862de", "#de62a8",
                  "#de6275", "#8f0b0b", "#8f4d0b", "#778f0b", "#0b8f47",
                  "#0b8f84", "#0b548f", "#4f0b8f", "#860b8f", "#8f0b51"]
    colorindex = 0
    for i in range(num_vertices):
        if i == 0:
            plt.scatter(data.x[i], data.y[i], c='g', s=50)
            plt.text(data.x[i], data.y[i], "depot", fontsize=12)
        else:
            plt.scatter(data.x[i], data.y[i], c='b', s=50)
            plt.text(data.x[i], data.y[i], f'{i}', fontsize=12)
    for route in trees:
        plotSubroute(route, data, color=colorslist[colorindex])
        colorindex += 1
        plt.savefig(directory + filename, dpi=300)
    plt.close()


# Implementation of the main experiment
def solveCVRP(save_graphs=True):
    VRP = [("A-n33-k6", 742), ("A-n36-k5", 799), ("A-n54-k7", 1167), ("B-n31-k5", 672), ("B-n34-k5", 788),
           ("B-n45-k5", 751), ("B-n50-k7", 741), ("B-n57-k7", 1153), ("E-n23-k3", 569), ("E-n33-k4", 835),
           ("F-n45-k4", 724), ("M-n101-k10", 820), ("M-n121-k7", 1034), ("P-n16-k8", 450), ("P-n19-k2", 212)]
    directory = "dataset/"
    for problem, optimal_distance in VRP:
        print("Solving %s using CWS-DTSA" % problem)
        print("-" * 100)
        print("CVRP: %s" % problem)
        print("Optimal Solution: %.2f" % optimal_distance)
        file = read_TSPLIB_CVRP(directory, problem)
        vertices = []
        for vertex in file.coordinate_points:
            vertex = str(vertex)[1:-1]
            vertex = re.sub(", ", " ", vertex)
            vertex_values = vertex.split()
            vertices.append((float(vertex_values[0]), float(vertex_values[1])))
        x, y = zip(*vertices)
        data = pd.DataFrame({"x": x, "y": y})
        num_vertices = len(vertices)
        demand_list = file.customer_demands
        distance_matrix = file.distance_matrix
        print(f'Depot coordinates: {vertices[0]}')
        print("Vertices:")
        print(f'\t{"Vertex" : ^6}{"Coordinate" : ^15}{"Demand" : ^20}')
        for i in range(num_vertices):
            print(f'\t{i : ^6}{str(vertices[i]) : ^15}{str(demand_list[i]) : ^20}')
        population_size = 10            # Control parameter
        search_tendency = 0.1           # Control parameter
        final_distances = []
        errors = []
        print("Results:")
        print(f'\t{"Trial" : ^5}{"Distance" : ^15}{"Relative Error (%)" : ^20}')
        start = time.time()
        for trial in range(30):
            CWS = paessens_savings_init(D=file.distance_matrix, d=file.customer_demands,
                                        C=file.capacity_constraint, L=None)
            CWS_route = [node for node in CWS if node != 0]
            solutions, distances, best_tree_index = DTSA(CWS_route, distance_matrix, num_vertices, population_size,
                                                         search_tendency, file)
            final_route = two_opt(routeToSubroute(solutions[best_tree_index], file), distance_matrix)
            final_distance = round(getRouteCost(final_route, file.distance_matrix), 2)
            error = abs((optimal_distance - final_distance)) / optimal_distance * 100
            final_distances.append(final_distance)
            errors.append(error)
            print(f'\t{trial + 1 : ^5}{final_distance : ^15.2f}{error : ^20.2f}')
            if save_graphs:
                plotRoute(final_route, num_vertices, data, "results/",
                          f'{problem}_trial{trial + 1}.png')

        end = time.time()
        elapsed_time = end - start

        # Sort the final distances list in ascending order
        sort_final_distances = sorted(final_distances)

        # Split the distances list into those below and above the optimal distance
        distances_below_optimal = [d for d in sort_final_distances if d < optimal_distance]
        distances_above_optimal = [d for d in sort_final_distances if d >= optimal_distance]

        # Calculate the worst distance below the optimal distance
        if distances_below_optimal:
            worst_below_optimal = abs(min(distances_below_optimal) - optimal_distance) / optimal_distance
        else:
            worst_below_optimal = float('inf')

        # Calculate the worst distance above the optimal distance
        if distances_above_optimal:
            worst_above_optimal = abs(max(distances_above_optimal) - optimal_distance) / optimal_distance
        else:
            worst_above_optimal = float('inf')

        # Find the best distance above the optimal distance
        if distances_above_optimal:
            best = min(distances_above_optimal)
        else:
            best = max(distances_below_optimal)

        # Choose the worst distance based on whether the worst above the optimal distance is worse than the worst
        # below the optimal distance
        if worst_above_optimal > worst_below_optimal:
            if not distances_above_optimal:
                worst = min(distances_below_optimal)
            else:
                worst = max(distances_above_optimal)
        else:
            worst = min(distances_below_optimal) if distances_below_optimal else max(distances_above_optimal)

        mean, stdev, error = np.mean(final_distances), np.std(final_distances), np.mean(errors)
        print("Summary:")
        print(f'\tPopulation Size: {population_size}\n\tSearch Tendency: {search_tendency}\n'
              f'\tRuntime (in seconds): {elapsed_time}')
        print(f'\t\t{"Best" : ^5}\t&{"Worst" : ^15}\t&{"Mean" : ^20}\t&{"SD" : ^25}\t&{"Error" : ^30}')
        print(f'\t\t{best : ^5}\t&{worst : ^15}\t&{mean : ^20.2f}\t&{stdev : ^25.2f}\t&{error : ^30.2f}\n')


# Driver program
solveCVRP()
