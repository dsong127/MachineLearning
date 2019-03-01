import sys
import random
from copy import deepcopy
#from queue import PriorityQueue

def main():
    input = sys.argv[1].replace('b', '0')
    input_list = list(map(int, input))

    p = Puzzle(input_list)
    p.solve_a_star("h2")


class Puzzle():
    def __init__(self, input):
        # 0 = b
        self.state_start = [input[i:i + 3] for i in range(0, 9, 3)]
        self.state = self.state_start
        self.state_goal = [[1,2,3], [4,5,6], [7,8,0]]

    def check_solved(self, state):
        return state == self.state_goal

    def get_blank_index(self, state):
        for r_idx, row in enumerate(state):
            for c_idx, e in enumerate(row):
                if e == 0:
                    row = r_idx
                    col = c_idx

        return row, col

    def available_moves(self, state):
        available_moves = []
        b_row, b_column = self.get_blank_index(state)

        if b_row == 0:
            available_moves.append("down")
        elif b_row == 1:
            available_moves.extend(("up", "down"))
        elif b_row == 2:
            available_moves.append("up")

        if b_column == 0:
            available_moves.append("right")
        elif b_column == 1:
            available_moves.extend(("left", "right"))
        elif b_column == 2:
            available_moves.append("left")

        random.shuffle(available_moves)

        return available_moves, b_row, b_column
    
    def move(self, state, direction):
        move_list, b_row, b_column = self.available_moves(state)

        new_state = deepcopy(state)

        if direction == "left":
            tile_to_move = state[b_row][b_column - 1]
            new_state[b_row][b_column] = tile_to_move
            new_state[b_row][b_column - 1] = 0
        elif direction == "right":
            tile_to_move = state[b_row][b_column + 1]
            new_state[b_row][b_column] = tile_to_move
            new_state[b_row][b_column + 1] = 0
        elif direction == "up":
            tile_to_move = state[b_row - 1][b_column]
            new_state[b_row][b_column] = tile_to_move
            new_state[b_row - 1][b_column] = 0
        elif direction == "down":
            tile_to_move = state[b_row+ 1][b_column]
            new_state[b_row][b_column] = tile_to_move
            new_state[b_row + 1][b_column] = 0

        return new_state

    def print_state(self):
        # Display the state of the board in the format "b12 345 678"
        str_state = []

        # Iterate through all the tiles
        for row in self.state:
            for element in row:
                if element == 0:
                    str_state.append("b")
                else:
                    str_state.append(str(element))

        # Print out the resulting state
        print("".join(str_state[0:3]), "".join(str_state[3:6]), "".join(str_state[6:9]))

    def pretty_print_state(self, state):
        print("\nCurrent State")
        for row in (state):
            print("-" * 13)
            print("| {} | {} | {} |".format(*row))

    def pretty_print_solution(self, solution_path):
        # Display the solution path in an aesthically pleasing manner
        try:
            # Solution path is in reverse order
            for depth, state in enumerate(solution_path[::-1]):
                if depth == 0:
                    print("\nStarting State")

                elif depth == (len(solution_path) - 2):
                    print("\nGOAL!!!!!!!!!")
                    for row_num, row in enumerate(state[0]):
                        print("-" * 13)
                        print("| {} | {} | {} |".format(*row))

                    print("\n")
                    break
                else:
                    print("\nDepth:", depth)
                for row_num, row in enumerate(state[0]):
                    print("-" * 13)
                    print("| {} | {} | {} |".format(*row))
        except:
            print("No Solution Found")

    def misplaced_heuristic(self, state):
        state_flat = sum(state, [])
        goal_flat = sum(self.state_goal, [])
        heuristic = 0

        for i in range(0,3):
            for j in range(0,3):
                if state_flat != goal_flat:
                    heuristic += 1

        return heuristic

    def manhattan_heuristic(self, state):
        # Calculates and return the h2 heuristic for a given state
        # The h2 hueristic for the eight puzzle is defined as the sum of the Manhattan distances of all the tiles
        # Manhattan distance is the sum of the absolute value of the x and y difference of the current tile position from its goal state position

        state_dict = {}
        goal_dict = {}
        heuristic = 0

        # Create dictionaries of the current state and goal state
        for row_index, row in enumerate(state):
            for col_index, element in enumerate(row):
                state_dict[element] = (row_index, col_index)

        for row_index, row in enumerate(self.goal_state):
            for col_index, element in enumerate(row):
                goal_dict[element] = (row_index, col_index)

        for tile, position in state_dict.items():
            # Do not count the distance of the blank
            if tile == 0:
                pass
            else:
                # Calculate heuristic as the Manhattan distance
                goal_position = goal_dict[tile]
                heuristic += (abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1]))

        return heuristic

    def calculate_f(self, node_depth, state, heuristic):
        # Returns the total cost of a state given its depth and the heuristic
        # Total cost in a-star is path cost plus heuristic. The path cost in this case is depth, or the number of moves from the start state to the current state because all moves have the same cost

        if heuristic == "h2":
            return node_depth + self.misplaced_heuristic(state)
        elif heuristic == "h1":
            return node_depth + self.manhattan_heuristic(state)

    def solve_a_star(self, heuristic, nodes_max = 10000, print_solution = True):
        nb_nodes = 0
        state_starting = deepcopy(self.state_start)
        current_state = deepcopy(self.state_start)

        # Nodes
        open_list = {}
        close_list = {}
        node_idx = 0
        current_depth = 0
        # Add starting state to both lists
        open_list[node_idx] = {"state": current_state, "depth": 0, "parent": "root", "action": "none", "f": self.calculate_f(0, current_state, heuristic)}
        close_list[node_idx] = {"state": current_state, "depth": 0, "parent": "root", "action": "none", "f": self.calculate_f(0, current_state, heuristic)}

        # Que will act as priority queue, storing nodes that have yet to expand
        que = [(0, open_list[0]["f"])]

        # Attempt to find a solution with a limit on how many nodes can be expanded
        while node_idx <= nodes_max:
            # Depth of current state
            for i, node in close_list.items():
                if node["state"] == current_state:
                    current_depth = node["depth"]

            # Get avaialble moves for current state
            moves = self.available_moves(current_state)

            for move in moves:
                if node_idx >= nodes_max:
                    self.nb_nodes_generated = nodes_max
                    break

                new_state = self.move(current_state, move)
                new_state_parent = deepcopy(current_state)

                # Check if the new state has already been expanded
                repeat = False
                for node in close_list.values():
                    if node["state"] == new_state:
                        if node["parent"] == new_state_parent:
                            repeat = True

                # If the new state has already been expanded before, move on to expanding nodes in open_list
                if repeat:
                    continue
                # Else we need to add the new state to open_list and the queue
                else:
                    node_idx += 1
                    current_depth += 1

                    f_new_state = self.calculate_f(current_depth, new_state, heuristic)
                    que.append((node_idx, f_new_state))
                    open_list[node_idx] = {"state": new_state, "depth": current_depth, "parent": new_state_parent, "action": move, "f": f_new_state}
                    print(open_list)

            # Sort by f cost
            que = sorted(que, key=lambda x: x[1])
            print(que)
            if node_idx <= nodes_max:
                best_node = que.pop(0)
                best_node_idx = best_node[0]
                best_node_state = open_list[best_node_idx]["state"]
                current_state = best_node_state
                print(best_node_state)


                closed_node = open_list.pop(best_node_idx)
                close_list[best_node_idx] = closed_node

                if self.check_solved(best_node_state):
                    self.expanded_nodes = close_list
                    self.frontier_nodes = open_list
                    self.nb_nodes_generated = node_idx + 1

                    self.success(close_list, node_idx, print_solution)
                    break

    def success(self, node_dict, num_nodes_generated, print_solution=True):
        # Once the solution has been found, prints the solution path and the length of the solution path
        if len(node_dict) >= 1:

            # Find the final node
            for node_num, node in node_dict.items():
                if node["state"] == self.goal_state:
                    final_node = node_dict[node_num]
                    break

            # Generate the solution path from the final node to the start node
            solution_path = self.generate_solution_path(final_node, node_dict,
                                                        path=[([[0, 1, 2], [3, 4, 5], [6, 7, 8]], "goal")])
            solution_length = len(solution_path) - 2

        else:
            solution_path = []
            solution_length = 0

        self.solution_path = solution_path

        if print_solution:
            # Display the length of solution and solution path
            print("Solution found!")
            print("Solution Length: ", solution_length)

            # The solution path goes from final to start node. To display sequence of actions, reverse the solution path
            print("Solution Path", list(map(lambda x: x[1], solution_path[::-1])))
            print("Total nodes generated:", num_nodes_generated)

    def generate_solution_path(self, node, node_dict, path):
        # Return the solution path for display from final (goal) state to starting state
        # If the node is the root, return the path
        if node["parent"] == "root":
            # If root is found, add the node and then return
            path.append((node["state"], node["action"]))
            return path

        else:
            # If the node is not the root, add the state and action to the solution path
            state = node["state"]
            parent_state = node["parent"]
            action = node["action"]
            path.append((state, action))

            # Find the parent of the node and recurse
            for node_num, expanded_node in node_dict.items():
                if expanded_node["state"] == parent_state:
                    return self.generate_solution_path(expanded_node, node_dict, path)

    def pretty_print_solution(self, solution_path):
        # Display the solution path in an aesthically pleasing manner
        try:
            # Solution path is in reverse order
            for depth, state in enumerate(solution_path[::-1]):
                if depth == 0:
                    print("\nStarting State")

                elif depth == (len(solution_path) - 2):
                    print("\nGOAL!!!!!!!!!")
                    for row_num, row in enumerate(state[0]):
                        print("-" * 13)
                        print("| {} | {} | {} |".format(*row))

                    print("\n")
                    break
                else:
                    print("\nDepth:", depth)
                for row_num, row in enumerate(state[0]):
                    print("-" * 13)
                    print("| {} | {} | {} |".format(*row))
        except:
            print("No Solution Found")

if __name__ == '__main__':
    main()