import numpy as np
import random

class QLearning:

    def __init__(self, reward_matrix, Q_table, num_state, num_action, err_array):
        self.gamma = 0.75 # Discount factor 
        self.alpha = 0.95 # Learning rate 
        self.R = reward_matrix
        self.Q_value = Q_table
        self.num_state = num_state
        self.num_action = num_action
        self.err_array = err_array
        self.QlearnCalculate()

    def QlearnCalculate(self):
        print("Q Learning Calculating....")
        for n in range(1, 1000):
            init_state = random.randint(0, self.num_state - 1)   
            state_current = init_state 
            while True:
                playable_actions = []
                for j in range(self.num_action):
                    if self.R[state_current, j] > 0:
                        playable_actions.append(j)
                action = np.random.choice(playable_actions)   
                state_next = action 
                TD = self.R[state_current, action] + self.gamma * max(self.Q_value[state_next, :]) - self.Q_value[state_current, action]
                self.Q_value[state_current, action] = self.Q_value[state_current, action] + self.alpha * TD
                state_current = state_next
                if self.err_array[state_current] <= min(self.err_array):
                    break
        print("Q Learning Calculated")

def cal_F(u, v, k, f, Am):
    """ 
    Calculate base element of hopf oscillator
    Input:
    Output:
    """
    return k * (Am * Am - u ** 2 - v ** 2) * u - 2 * np.pi * f * v,  k * (Am * Am - u ** 2 - v ** 2) * v + 2 * np.pi * f * u

def cal_P_head(post_u, post_v, epsilon, psi):
    """
    """
    return epsilon * (post_v * np.cos(psi) - post_u * np.sin(psi))

def cal_P_tail(pre_u, pre_v, epsilon, psi):
    """
    """
    return epsilon * (pre_u * np.sin(psi) + pre_v * np.cos(psi))

def cal_P_body(pre_u, pre_v, post_u, post_v, epsilon, psi):
    """
    """
    return epsilon * (pre_u * np.sin(psi) + pre_v * np.cos(psi) - post_u * np.sin(psi) + post_v * np.cos(psi))

endtime = 20
step = 0.01

def CPG_Calculate(k_in):
    array_u = np.zeros([1, 16])
    array_v = np.zeros([1, 16])
    array_theta = np.zeros([1, 16])    
    array_time = [0] 
    array_psi_r = [0]

    # initialize parameter
    time = 0
    Ax = 1
    k = k_in
    f = 1
    check = 1
    epsilon = 0.8
    psi = -np.pi / 3

    # initial state
    array_u[0][0] = 0
    array_v[0][0] = 0.001

    for idx in range(1, int(endtime / step)):
        time = time + 0.01
        state_u = []
        state_v = []
        state_theta = []

        array_theta = np.append(array_theta, np.zeros([1, 16]), axis=0)
        array_u = np.append(array_u, np.zeros([1, 16]), axis=0)
        array_v = np.append(array_v, np.zeros([1, 16]), axis=0)

        for i in range(16):
            A = 1
            # compute the base element
            F_u, F_v = cal_F(array_u[idx - 1][i], array_v[idx - 1][i], k, f, Ax)
            # compute new state of ith CPG at time idx*step with newton approximation
            new_u = F_u * step + array_u[idx - 1][i]
            if i == 0:
                new_v = (F_v + cal_P_head(array_u[idx - 1, 1], array_v[idx - 1, 1], epsilon, psi)) * step + array_v[idx - 1][i]
            elif i == 15:
                new_v = (F_v + cal_P_tail(array_u[idx - 1, 14], array_v[idx - 1, 14], epsilon, psi)) * step + array_v[idx - 1][i]
            else: 
                new_v = (F_v + cal_P_body(array_u[idx - 1, i - 1], array_v[idx - 1, i - 1], array_u[idx - 1, i + 1], array_v[idx - 1, i + 1], epsilon, psi)) * step + array_v[idx - 1][i]
            new_theta = A * new_u
            # create new state vector
            state_u.append(new_u)
            state_v.append(new_v)
            state_theta.append(new_theta)
        # add new state vector to original placeholder
        array_u[idx] = array_u[idx] + state_u
        array_v[idx] = array_v[idx] + state_v
        array_theta[idx] = array_theta[idx] + state_theta
        if array_u[idx, 15] > 0.2:
            if check == 1:
                check = 0
                get_time = time
    error = np.absolute(1 - max(array_u[:, 5]))
    return get_time, error

if __name__ == '__main__':
    size = 150
    time_array = []
    error_array = []
    k_array = []

    for k in range(1, size + 1):
        k_array.append(k)
        ti, err = CPG_Calculate(k)
        time_array.append(ti)
        error_array.append(err)
        print("k:", k, "Time:", ti, "Error:", err)

    # Q-Learning 
    # Create Reward and Q table
    reward = np.zeros([size, size])
    Q_table = np.zeros([size, size])

    for i in range(0, size):  # state
        for j in range(0, size): # action
            if j == i:
                reward[i, j] = 0
            elif j != i :
                if error_array[j] > error_array[i]:
                    reward[i, j] = 0
                elif error_array[j] <= error_array[i]:
                    if error_array[j] <= min(error_array):
                        reward[i, j] = 100
                    else:
                        reward[i, j] = 10

    for n in range(0, size):
        for m in range(0, size):
            if m == n:
                reward[n, m] = 0
            elif m != n :
                if time_array[m] > time_array[n]:
                    reward[n, m] = reward[n, m] + 0
                elif time_array[m] <= time_array[n]:
                    reward[n, m] = reward[n, m] + 10

    qlearn = QLearning(reward_matrix=reward, Q_table=Q_table, num_state=size, num_action=size, err_array=error_array)
    print("Q table:")
    print(Q_table)
    print("Reward:")
    print(reward)

    max_value = []
    for n in range(0, size):
        max_value.append(max(Q_table[:, n]))

    cov_rate = np.argmax(max_value) + 1

    print("Q value max:", max(max_value))
    print("Rate k:", cov_rate)
