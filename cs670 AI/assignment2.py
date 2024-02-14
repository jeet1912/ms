
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


#Test1
state1 = [True, True, True, True]

#Test2 
state2 = [False, False, False, False]

#Test3
states3 = [True, False, True, False]

states = [state1, state2, states3]

for state in states:
    print(state_description(state))
    print('------------------------------------')
