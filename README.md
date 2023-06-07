# Parallel Paessens' Clarke-Wright Savings &ndash; Discrete Tree-Seed Algorithm for Optimization of the Capacitated Vehicle Routing benchmark Problems
Optimizing Capacitated Vehicle Routing Problem through Modified Discrete Tree-Seed Algorithm with Parallel Paessens' Clarke-Wright Heuristic

In this study, we proposed the use of the modified Discrete Tree-Seed Algorithm with the Parallel Paessens' Clarke-Wright Savings (CWS-DTSA) in optimizing Capacitated Vehicle Routing benchmark problems (CVRP). The control parameters considered are the search tendency *ST* and population size *N*. The additional parameters in the algorithm include the *route shape* parameter λ and *correction term* μ which were used to solve for the savings matrix of the Clarke-Wright Savings algorithm.

For each benchmark problem of the CVRP, the fleet size is **homogeneous** which means that the vehicle capacity is the same for all vehicles. Various number of customers were considered ranging from 16 to 121 customers with different vehicle capacities *Q* and demands *q<sub>i</sub>*. Hence, each parameter is considered as a single-dimension in a multi-dimensional search space such that the other dimensions in the search space corresponds to the several problem-specific parameters such as the number of nodes, the number of vehicles, the capacity constraints, the distance matrix, and the savings matrix, among others.

In the algorithm, the nodes are denoted by tuples and these array of tuples are considered to be the dimension of the CVRP problem. The search space, on the other hand, represents the set of all possible combination of routes that satisfy the constraints of the CVRP such as the capacity limits of the vehicles and the requirement that each customer is visited exactly once.

The trees (solutions) and seeds (feasible solutions) are a set of possible routes wherein the dimension is the number of nodes while the search space is the combination of routes. The trees are first initialized with the help of the CWS algorithm and the random permutations of the nodes. Then, the algorithm iterates over all the trees in the population while generating seeds to improve the trees. The maximum number of function evaluation was used to be the terminating criterion.

The proposed CWS-DTSA showed promising results in optimizing benchmark CVRPs. The algorithm was able to achieve optimal solutions with an acceptable percentage deviation and runtime. The parameter analysis showed that population size *N* has a greater effect on the algorithm's performance compared to the search tendency *ST*. Additionally, the study found that as the number of vertices and population size increase, so does the algorithm's runtime. Computational results also show that the algorithm has a satisfactory performance obtaining 53.33% of optimal or quality solutions, with 46.67% deviated solutions compared to the Genetic Algorithm which has 66.67% optimum achievements, 6.67% new solutions, and 26.67% deviated solutions. Furthermore, the proposed algorithm has better performance than the original DTSA which uses the Nearest Neighbor algorithm and the Clarke-Wright Savings heuristic if used solely.

Based on the findings, we recommend further exploration of the proposed algorithm's potential in solving more complex optimization problems. Specifically, future studies may focus on enhancing the algorithm's consistency in achieving optimal solutions by adjusting the maximum number of function evaluations. Additionally, the algorithm's performance in real-world scenarios could also be explored further. Overall, the results of this study suggest that the proposed algorithm is a promising approach in solving the Capacitated Vehicle Routing benchmark problems and has the potential for broader application in various optimization problems.

## Hardware and Software Specifications

The experiments were conducted on an Aspire A315-41G laptop equipped
with AMD Ryzen 3 2200U with Radeon Vega Mobile Gfx @ 2.50GHz and 12.0 Gigabytes
installed RAM. The computer was running on the Windows 10 operating system. The use
of this hardware allowed us to obtain quality solutions within a reasonable amount of time.

The CWS-DTSA was built using Python 3.10.6 libraries such as Pandas, Numpy, Matplotlib, and
VeRyPy.

* **Pandas** is an open source Python package used for data analysis and machine learning tasks. It is built on top of another package named Numpy, which provides support for multi-dimensional arrays.

     – **DataFrame** is a 2-dimensional data structure with columns of potentially different types. In this study, DataFrame was used to assign the (x, y) coordinate
values.

* **Numpy** is used for the scientific computing of the mean and standard deviation of the n-dimensional array.

* **Matplotlib** is a comprehensive library for creating static, animated, and interactive visualization in python. The figures of this study will be generated using Matplotlib.

* **VeRyPy** is an easy to use library of classical CVRP algorithms with symmetric distances. Compared to the existing heuristic and metaheuristic open source VRP libraries, it focuses on re-usability of the code and in faithful recreation of the original algorithms.

## The Capacitated Vehicle Routing Problem
CVRP is defined as a graph *G = (V, E)* which consists of nodes *V = {0, 1, ..., N}* and an edge set *E = {(i, j) : i, j ∈ V}*. Vertex 0 represents the depot and the other nodes {1, 2, ..., N} represents the customers who have specific demands *q<sub>i</sub>*, where *i = {1, 2, ..., N}*, to be delivered. The travel cost between node *i* and *j* is defined by *c<sub>i, j</sub> > 0*. For a single depot, a set of homogeneous vehicles *K* with capacity limit *Q* depart from and return to. If vehicle *k* travels from customer *i* to customer *j* directly, *X<sub>ij</sub><sup>k</sup> = 1* otherwise, *X<sub>ij</sub><sup>k</sup> = 0*. Hence, the objective function of CVRP is *f = Σ<sup>N</sup><sub>i = 0</sub> Σ<sup>N</sup><sub>j = 0</sub> Σ<sup>K</sup><sub>k = 0</sub> C<sub>ij</sub>X<sub>ij</sub><sup>k</sup>*, which minimizes the total distance traveled by the vehicles, is subject to sme constraints such as (i) customers can only be serviced by one vehicle, (ii) the total demand of all customers on any route must not exceed the vehicle capacity *Q*, and (iii) all routes must start and finish at the same depot after servicing the customers. 

A solution of CVRP is feasible if all routes satisfy the capacity constraint and no customer is visited more than once. The number of vehicles *K* that serve the customers will be a decision variable in this study such that we are not limiting the routes by the number of vehicles. Hence, expect that there will be routes with more or less vehicles than the expected number of vehicles.

## The Discrete Tree-Seed Algorithm
The DTSA is a stochastic, nature-inspired, metaheuristic algorithm that mimics the relationship between trees and seeds. The reason for this miicry is that as seeds grow, they replace the tree where it came from. The algorithm was a version of the original TSA. TSA was used for continuous optimization problems while DTSA was used to solve symmetric TSPs.

The trees and seeds represent the possible tours for the CVRP problem. These are represented by various combinations of routes. In addition, the number of dimensions *D*, is the number of the vertices excluding the depot. For example, if the number of vertices/customers is 6, then *D = 6* which also shows that each customer *v<sub>i</sub>* is a 6-tuple with D = V = (*v<sub>1</sub>v<sub>2</sub>v<sub>3</sub>v<sub>4</sub>v<sub>5</sub>v<sub>6</sub>*), and the search space can be represented as, *I = I<sub>1</sub> &#215; I<sub>2</sub> &#215; I<sub>3</sub> &#215; I<sub>4</sub> &#215; I<sub>5</sub> &#215; I<sub>6</sub>* where each search space *I<sub>n</sub>* consists of all possible combinations of valid trees and seeds for the given CVRP instance. Specifically, the search space for the CVRP is determined by all possible ways of dividing the customer nodes into feasible routes. The search space size depends on the customer nodes, the vehicle capacity, and other constraints. For instance, if there are 3 customer nodes, a single vehicle with a capacity of 5, and no distance constraints, the search space might be:

Number of possible routes for one vehicle: *3C1 + 3C2 + 3C3 = 7*
Number of possible node sequences for each route: *3! = 6*
Total number of possible solutions: *I = 7 &#215; 6 = 42*

Since the trees and seeds are used as a permutation of the customers, each tree and seed is unique and can only represent one CVRP solution. For instance, if our CVRP consists of 6 customers, one of the trees that can be used is *t = v<sub>0</sub>v<sub>6</sub>v<sub>2</sub>v<sub>3</sub>v<sub>5</sub>v<sub>1</sub>v<sub>4</sub>v<sub>0</sub>*. To improve this initial tree into a desired solution, the seeds are created using transformation operators with details about *Q* and *q<sub>i<\sub>*. Hence, one possible seed is, *s = v<sub>0</sub>v<sub>6</sub>v<sub>3</sub>v<sub>0</sub>v<sub>2</sub>v<sub>5</sub>v<sub>0</sub>v<sub>1</sub>v<sub>4</sub>v<sub>0</sub>*, such that CVRP constraints were followed.

## Clarke-Wright Savings Algorithm
The Clarke and Wright "savings" algorithm (CWS) is one of the best-known and simplest approaches to solve the VRP. Since the original formula for the savings matrix of the CWS does not guarantee optimal nor near-optimal results, we used Paessens' savings formula that is,

s(i, j) = [d(v0 , i) + d(v0 , j) − λd(i, j)] + [µ|d(v0 , i) − d(v0 , j)|]

where {λ ∈ R : 0 < λ ≤ 2} and {µ ∈ R : 0 < µ ≤ 2}. The values λ and µ greatly affect the routes generated by the CWS. The route shape parameter λ controls the shape of the routes. By adjusting this parameter, it is possible to generate routes that are more or less efficient. For instance, if λ is set to a low value, the total distance traveled is minimized so as the number of vehicles. Otherwise, it will generate routes that are more compact and cover a smaller area. The correction term µ, on the other hand, penalizes the savings obtained from combining customers with large demands that exceed the capacity of the vehicle such that if the combined demand of customers *i* and *j* are high, the greater the penalty and makes it less likely for these nodes to be combined into the same route.

## Initialization
The DTSA starts with the intialization of a population of trees. The first tree is created by using CWS while the other trees are created as random permutations of the vertices of an instance of a VRP.

## Transformation Operators
The seeds are generated using transformation operators swap, shift, and symmetry.

## 2-Opt heuristic
The best tree generated by the CWS-DTSA are the input for the 2-Opt heuristic and will be further improved to get to the optimal solution.

## Sub Route Modifier for Capacity Constraint
Since the final output of the 2-Opt heuristic is a route divided by the depots, this route is decoded into a list of lists for simplicity.

## How to Use CWS-DTSA?
Input the .VRP file and the optimal distance in the VRP list of tuples. The main algorithm for solving the CVRP includes the Parallel Paessens CWS heuristic, DTSA, and 2-Opt Heuristic. The DTSA will start by initializing the population_size of the trees such that one of the trees is an output of CWS, and the tree with the lowest travel cost is determined as the best tree among the population. Then, the algorithm iterates over the population of trees continuously until the maximum number of function evaluation maxFE is met. 

For each tree being evaluated, transformation operators are used to generate a population of seeds, then the least-cost seed is identified as the best seed. The current tree and the best seed are then compared during the iteration. If the seed has a lower-cost than the current tree, then the seed will replace the current tree. Otherwise, the algorithm will continue to the next tree without considering the replacement of the current tree.
     
Overall, to use the Clarke-Wright Savings - Discrete Tree-Seed Algorithm, you just have to input the .VRP file and the optimal distance in the list of tuples and let the program do the work.

