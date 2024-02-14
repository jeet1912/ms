
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
    invalid_states = [[False, True, True, False], [False, False, True, True], [False, True, True, True], [True, False, False, True], [True, False, False, False], [True, True, False, False]] 
    
    if state in invalid_states:
        validity = False

    return validity

#function to get the subsequent state
def get_possible_moves(state):
    
    man, lion, goat, grass = state
    possible_moves = []
    
    """
    possible_moves.append([not man, lion, goat, grass])
    possible_moves.append([not man, not lion, goat, grass])
    possible_moves.append([not man, lion, not goat, grass])
    possible_moves.append([not man, lion, goat, not grass])
    """
    
    possible_moves = [[not state[j] if (j == 0 or i == j) else state[j] for j in range(4)] for i in range(4)]

    return [move for move in possible_moves if is_valid(move)]

#Test1
state = [False, False, True, False]
print(get_possible_moves(state))