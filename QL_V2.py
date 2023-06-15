import numpy as np
import random


if __name__ == '__main__':
    print("Q Learning Start")
    # print( np.argmax(Q[0,:]) )
    # Init statte - action 
    reward_state_action = np.array([[0,0,0,0,1,0],
                                    [0,0,0,1,0,100],
                                    [0,0,0,1,0,0],
                                    [0,1,1,0,1,0],
                                    [1,0,0,1,0,100],
                                    [0,1,0,0,1,100]
                                   ])
    Q_value = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]
                       ])    
    goal_state = 5
    epsilon = 0.5
    decay_epsilon = 0.95
    # Init Q table 
    gamma = 0.75 # Discount factor 
    alpha = 0.95 # Learning rate 
    numState = 6
    numAction = 6
    R = reward_state_action
    # Outer loop
    for n in range(1,1000):
        init_state = random.randint(0, numState-1)
        # Model 
        state_current = init_state
        epsilon *= decay_epsilon
        while True:
            # Inner loop
            playable_actions = []
            # Iterate through the new rewards matrix and get the actions > 0
            for j in range(numAction):
                if reward_state_action[state_current, j] > 0:
                    playable_actions.append(j)
            if np.random.random() < epsilon or np.sum(Q_value[state_current, :]) == 0:
                action = np.random.choice(playable_actions)
            else:
                action = np.argmax(Q_value[state_current, :])


            # state_next = model(action)
            state_next = action
            # Q-Learning 
            TD = R[state_current, action] + gamma * max(Q_value[state_next, :]) - Q_value[state_current, action]
            Q_value[state_current, action] = Q_value[state_current, action] + alpha * TD
            # Update Q value table

            state_current = state_next
            if state_current == goal_state:
                break
    
    print("Q Learning End")

    print(Q_value)
