
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

#Test1

state1 = [False, False, False, False]
state2 = [True, True, True, True]
state3 = [False, True, False, True]
state4 = [True, False, True, False]
state5 = [False, True, True, False]

state = [state1, state2, state3, state4, state5]

for s in state:
    print(is_valid(s))