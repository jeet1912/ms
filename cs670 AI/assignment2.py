from collections import deque
import datetime

#False = starting side and True = target side.

#dictionary to store the entities
entities = {0: 'Man', 1:'Lion', 2:'Goat', 3:'Grass'}

#Function to describe the state
def state_description(state):
    
    starting_side = []
    target_side = []
    
    for index, bool in enumerate(state):
        if bool == True:
            target_side.append(entities[index])
        
        else:
            starting_side.append(entities[index])

    state_description = 'starting side: ' + ', '.join(starting_side) + '\n' + 'target side: ' + ', '.join(target_side)
    
    return state_description 


#Function to check if the state is valid
def is_valid(state):

    validity = True
    if ((state[1] == state[2]) and (state[0] != state[1])) or ((state[2] == state[3]) and (state[0] != state[2])):
        validity = False

    return validity

#function to get the subsequent state
def get_possible_moves(state):
    
    #man, lion, goat, grass = state
    #possible_moves = []
    
    """
    possible_moves.append([not man, lion, goat, grass])
    possible_moves.append([not man, not lion, goat, grass])
    possible_moves.append([not man, lion, not goat, grass])
    possible_moves.append([not man, lion, goat, not grass])
    """
    
    possible_moves = [[not state[j] if (j == 0 or i == j) else state[j] for j in range(4)] for i in range(4)]

    return [move for move in possible_moves if is_valid(move)]



def bfs(start, goal):
    start = list(start)
    goal = list(goal)
    queue = deque([[start]])
    visited = set(tuple(start)) #since we always expand start node.
    while queue:
        path = queue.popleft()
        curr_state = path[-1]
        print('Visited states in BFS: ', len(visited))
        if curr_state == goal:
            return path
        for next_state in get_possible_moves(curr_state):
            if tuple(next_state) not in visited: 
                visited.add(tuple(next_state))  
                queue.append(path + [next_state])
    

def dfs(start,goal,path=[], visited=set()):
    start = list(start)
    goal = list(goal)
    path = path + [start]
    visited.add(tuple(start))
    if start == goal:
        return path
    for next_state in get_possible_moves(start):
        if tuple(next_state) not in visited:
            new_path = dfs(next_state, goal, path, visited)
            print('Visited states in DFS: ', len(visited))
            if new_path:
                return new_path

#start_time = datetime.datetime.now()        
start_state = (False, False, False, False)
goal_state = (True, True, True, True)
solution1 = bfs(start_state, goal_state)
solution2 = dfs(start_state, goal_state)
#end_time = datetime.datetime.now()
#print("Time taken in milliseconds: ", (end_time - start_time).total_seconds() * 1000)
if solution1:
    for step in solution1:
        print(state_description(step))
else:
    print("No solution found.")

if solution2:
    for step in solution2:
        print(state_description(step))
else:
    print("No solution found.")

