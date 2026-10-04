# Inspired by AIFCA Python code Version 0.9.17
# Original source: https://aipython.org
#
# Artificial Intelligence: Foundations of Computational Agents
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
# https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# Adapted and extended by Anna Trofimova
# Professional Teaching Fellow, School of Computer Science
# University of Auckland
# 2026
#
# Hill climbing (by Anna Trofimova) revised to the lectures' sideways-move rule,
# and minimax search added, by Di Zhao, 2026


from problem import Path
import heapq
import random
from csp import Constraint
from itertools import product

class FrontierPQ(object):
    """
    A frontier implemented as a priority queue (min-heap), storing (value, index, path) tuples, where:
    - value: the quantity to minimise (e.g., cost + heuristic)
    - index: a unique counter used to break ties (FIFO order)
    - path: a sequence of states representing the current path

    The priority queue returns the path with the smallest value first.
    """

    def __init__(self):
        """Constructs the frontier as an empty priority queue."""
        self.frontier_index = 0  # Unique counter to maintain insertion order
        self.frontierpq = []  # List to hold heap elements

    def add(self, path: Path, value: float) -> None:
        """
        Adds a path to the priority queue.

        Args:
            path: The path to be added.
            value: The priority value to minimise (e.g., cost or cost + heuristic).
        """
        self.frontier_index += 1
        # Use negative index to prioritise earlier-inserted paths in case of ties
        heapq.heappush(self.frontierpq, (value, -self.frontier_index, path))

    def pop(self) -> Path:
        """
        Removes and returns the path with the minimum value (highest priority).
        """
        _, _, path = heapq.heappop(self.frontierpq)
        return path

    def __repr__(self):
        """Returns a string representation of the frontier for debugging."""
        return str([(v, i, str(p)) for (v, i, p) in self.frontierpq])

    def __len__(self):
        """Returns the number of elements currently in the frontier."""
        return len(self.frontierpq)

    def __iter__(self):
        """Iterates over paths in the frontier (heap order, not sorted)."""
        for _, _, path in self.frontierpq:
            yield path


class Searcher:
    """
    A generic searcher for a graph search problem.

    Implements Depth-First Search (DFS) by default via a stack-based frontier.
    Subclasses override `initialize_frontier` and `add_to_frontier` for different strategies.
    """

    def __init__(self, problem):
        """
        Initializes the searcher with the given problem instance.

        Args:
            problem: An object representing the search problem.
        """
        self.problem = problem
        self.initialize_frontier()
        self.num_expanded = 0
        self.max_frontier_size = 0
        self.solution = None
        # Start search from the start node
        self.add_to_frontier(Path(problem.start_node()))

    def initialize_frontier(self):
        """Initializes the frontier as an empty stack (DFS default)."""
        self.frontier = []

    def is_empty(self):
        """Alternative check for frontier emptiness (used internally)."""
        return len(self.frontier) == 0

    def empty_frontier(self):
        """Checks whether the frontier is empty (external use)."""
        return len(self.frontier) == 0

    def add_to_frontier(self, path):
        """
        Adds a path to the frontier and updates the maximum frontier size.

        Args:
            path: A Path object representing the new path.
        """
        self.frontier.append(path)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))

    def search(self):
        """
        Performs a DFS-based search to find a goal.

        Returns:
            Path: A valid path to a goal state if found, else None.
        """
        while not self.empty_frontier():
            current_path = self.frontier.pop()  # DFS: LIFO
            self.num_expanded += 1

            if self.problem.is_goal(current_path.end()):
                self.solution = current_path
                return current_path

            # Expand and add neighbours to frontier
            for arc in reversed(list(self.problem.neighbors(current_path.end()))):
                new_path = Path(current_path, arc)
                self.add_to_frontier(new_path)

        return None  # No solution found


class BreadthFirstSearcher(Searcher):
    """
    Breadth-First Search (BFS) using a FIFO frontier.
    """

    def search(self):
        """
        Performs BFS until a goal is found or frontier is empty.

        Returns:
            Path: A valid path to a goal state if found, else None.
        """
        while not self.is_empty():
            current_path = self.frontier.pop(0)  # BFS: FIFO
            self.num_expanded += 1

            if self.problem.is_goal(current_path.end()):
                self.solution = current_path
                return current_path

            for arc in self.problem.neighbors(current_path.end()):
                new_path = Path(current_path, arc)
                self.add_to_frontier(new_path)

        return None


class AStarSearcher(Searcher):
    """
    A* Search using a priority queue frontier ordered by cost + heuristic.
    """

    def initialize_frontier(self):
        """Overrides default stack with priority queue."""
        self.frontier = FrontierPQ()

    def add_to_frontier(self, path: Path):
        """
        Adds path to the priority queue with f(n) = g(n) + h(n).
        """
        value = path.cost + self.problem.heuristic(path.end())
        self.frontier.add(path, value)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))


class UniformCostSearcher(Searcher):
    """
    Uniform Cost Search using a priority queue ordered by path cost g(n).
    """

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def add_to_frontier(self, path: Path):
        value = path.cost  # Priority is g(n)
        self.frontier.add(path, value)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))


class GreedySearcher(Searcher):
    """
    Greedy Best-First Search using a priority queue ordered by h(n).
    """

    def initialize_frontier(self):
        self.frontier = FrontierPQ()

    def add_to_frontier(self, path: Path):
        value = self.problem.heuristic(path.end())  # Priority is h(n)
        self.frontier.add(path, value)
        self.max_frontier_size = max(self.max_frontier_size, len(self.frontier))


class IterativeDeepeningSearcher(Searcher):
    """
    Iterative Deepening Search (IDS): DFS with increasing depth limits.
    """

    def search_upto(self, depth_limit):
        """
        Performs depth-limited DFS.

        Args:
            depth_limit: The maximum depth to search.

        Returns:
            Path: A valid path to the goal state if found, else None.
        """
        while not self.empty_frontier():
            current_path = self.frontier.pop()
            self.num_expanded += 1

            if self.problem.is_goal(current_path.end()):
                self.solution = current_path
                return current_path

            if len(list(current_path.nodes())) >= depth_limit:
                continue

            for arc in reversed(list(self.problem.neighbors(current_path.end()))):
                new_path = Path(current_path, arc)
                self.add_to_frontier(new_path)

        return None

    def search(self):
        """
        Performs Iterative Deepening Search by gradually increasing the depth limit.

        Returns:
            Path: A valid path to the goal state if found, else None.
        """
        depth = 0

        path = None
        while not path:
            self.num_expanded = 0
            self.solution = None
            self.max_frontier_size = 0
            depth += 1
            path = self.search_upto(depth)

            # If not found, reset frontier before next iteration
            if not path:
                self.initialize_frontier()
                self.add_to_frontier(Path(self.problem.start_node()))

        return path


class HillClimbingSearcher:
    """
    Hill climbing with sideways moves for a CSP (local search over complete assignments).

    The quantity to minimise is the number of violated constraints. At each step the searcher
    moves to a randomly chosen *improving* neighbour (one with fewer violated constraints).
    Only when no improving neighbour exists does it make a *sideways* move, to a randomly
    chosen neighbour with the same number of violated constraints; at most `side_moves`
    sideways moves may be made in a row (the count resets after every improving move).

    The search stops at a solution, or when it has no improving neighbour and may not
    (or cannot) move sideways.
    """

    def __init__(self, csp, side_moves=0):
        """
        Args:
            csp: A CSP instance.
            side_moves: The maximum number of consecutive sideways moves (0 = none).
        """
        self.csp = csp
        self.side_moves = side_moves
        self.steps = 0

    def random_assignment(self):
        """Returns a random complete assignment."""
        return {var: random.choice(var.domain) for var in self.csp.variables}

    def heuristic(self, assignment):
        """Returns the number of violated constraints (lower is better)."""
        violations = 0
        for con in self.csp.constraints:
            if con.can_evaluate(assignment):
                if not con.holds(assignment):
                    violations += 1
        return violations

    def get_neighbors(self, assignment):
        """Returns all assignments that differ from `assignment` in exactly one variable."""
        neighbors = []
        for var in self.csp.variables:
            for value in var.domain:
                if value != assignment[var]:
                    new_assign = assignment.copy()
                    new_assign[var] = value
                    neighbors.append(new_assign)
        return neighbors

    def search(self):
        """
        Runs hill climbing with sideways moves from a random complete assignment.

        Returns:
            (assignment, h): the final assignment and its number of violated
            constraints; h == 0 means a solution was found. `self.steps` holds the
            number of moves made.
        """
        self.steps = 0
        current = self.random_assignment()
        current_heur = self.heuristic(current)
        side_moves_left = self.side_moves

        while current_heur > 0:
            scored = [(self.heuristic(n), n) for n in self.get_neighbors(current)]
            better = [n for h, n in scored if h < current_heur]

            if better:                                  # an improving move
                next_state = random.choice(better)
                side_moves_left = self.side_moves       # reset after an improvement
            else:                                       # no improving neighbour
                equal = [n for h, n in scored if h == current_heur]
                if side_moves_left == 0 or not equal:
                    break                               # stuck
                next_state = random.choice(equal)       # a sideways move
                side_moves_left -= 1

            self.steps += 1
            current = next_state
            current_heur = self.heuristic(current)

        return current, current_heur


class MinimaxSearcher:
    """
    Minimax search on an explicit game tree (see GameTree in problem.py).

    MAX moves at the root and the players alternate level by level. The search follows
    Minimax-Decision, Max-Value and Min-Value from the lectures, and counts how many leaves
    it evaluates.
    """

    def __init__(self, tree):
        """
        Args:
            tree: A GameTree instance.
        """
        self.tree = tree
        self.num_leaves = 0     # leaves evaluated by the last call to search()

    def search(self):
        """
        Minimax-Decision: returns the move leading to the child of the root with the
        largest Min-Value.

        Returns:
            (value, move): the minimax value of the root, and the child of the root that
            achieves it (the first such child, in the tree's order).
        """
        self.num_leaves = 0
        best_value, best_move = float('-inf'), None
        for child in self.tree.children(self.tree.root):
            v = self.min_value(child)
            if v > best_value:
                best_value, best_move = v, child
        return best_value, best_move

    def max_value(self, node):
        """Returns the minimax value of `node`, a node where MAX is to move."""
        if self.tree.is_leaf(node):
            self.num_leaves += 1
            return self.tree.utility(node)
        v = float('-inf')
        for child in self.tree.children(node):
            v = max(v, self.min_value(child))
        return v

    def min_value(self, node):
        """Returns the minimax value of `node`, a node where MIN is to move."""
        if self.tree.is_leaf(node):
            self.num_leaves += 1
            return self.tree.utility(node)
        v = float('inf')
        for child in self.tree.children(node):
            v = min(v, self.max_value(child))
        return v
