import numpy as np
import random
import matplotlib.pyplot as plt

def cal_F(u, v, k, f, Am):
    return k * (Am * Am - u ** 2 - v ** 2) * u - 2 * np.pi * f * v, k * (Am * Am - u ** 2 - v ** 2) * v + 2 * np.pi * f * u

def cal_P_head(post_u, post_v, epsilon, psi):
    return epsilon * (post_v * np.cos(psi) - post_u * np.sin(psi))

def cal_P_tail(pre_u, pre_v, epsilon, psi):
    return epsilon * (pre_u * np.sin(psi) + pre_v * np.cos(psi))

def cal_P_body(pre_u, pre_v, post_u, post_v, epsilon, psi):
    return epsilon * (pre_u * np.sin(psi) + pre_v * np.cos(psi) - post_u * np.sin(psi) + post_v * np.cos(psi))

def CPG_Calculate(k_in):
    array_u = np.zeros([1, 16])
    array_v = np.zeros([1, 16])
    array_theta = np.zeros([1, 16])
    array_time = [0]
    array_psi_r = [0]
    endtime = 11
    step = 0.01
    time = 0
    Ax = 1
    k = k_in
    f = 1
    check = 1
    epsilon = 0.8
    psi = -np.pi / 3
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
            F_u, F_v = cal_F(array_u[idx - 1][i], array_v[idx - 1][i], k, f, Ax)
            new_u = F_u * step + array_u[idx - 1][i]

            if i == 0:
                new_v = (F_v + cal_P_head(array_u[idx - 1, 1], array_v[idx - 1, 1], epsilon, psi)) * step + array_v[idx - 1][i]
            elif i == 15:
                new_v = (F_v + cal_P_tail(array_u[idx - 1, 14], array_v[idx - 1, 14], epsilon, psi)) * step + array_v[idx - 1][i]
            else:
                new_v = (F_v + cal_P_body(array_u[idx - 1, i - 1], array_v[idx - 1, i - 1], array_u[idx - 1, i + 1], array_v[idx - 1, i + 1], epsilon, psi)) * step + array_v[idx - 1][i]

            new_theta = A * new_u
            state_u.append(new_u)
            state_v.append(new_v)
            state_theta.append(new_theta)

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

    reward = np.zeros([size, size])
    Q_value = np.zeros([size, size])

    time_array = []
    error_array = []
    k_array = []

    for k in range(1, size + 1):
        k_array.append(k)
        ti, err = CPG_Calculate(k)
        time_array.append(ti)
        error_array.append(err)
        with open("CPG_data.txt", "a") as f:
            f.write(str(k) + "\t" + str(ti) + "\t" + str(err) + "\n")
        print("k: ", k, "Time: ", ti, "Error: ", err)

    gamma = 0.75
    alpha = 0.95
    epsilon = 0.7
    R = reward
    num_state = size
    num_action = size
    add_reward = 0
    R = reward
    num_episodes = 3000
    arr_min_time = []
    arr_min_error = []
    arr_si = []

    print("Q Learning Calculating....")

    for n in range(0, num_episodes):
        init_state = random.randint(0, num_state - 1)
        state_current = init_state
        arr_si = []
        pre_ti = time_array[state_current]
        pre_err = error_array[state_current]

        if (n + 1) % 100 == 0:
            print("Episode {} of {}".format(n + 1, num_episodes))

        while True:
            ti_reward = 0
            err_reward = 0
            si = 100 * pre_err + 10 * pre_ti
            arr_si.append(si)

            arr_min_time.append(pre_ti)
            arr_min_error.append(pre_err)

            if np.random.random() < epsilon:
                action = random.randint(0, num_action - 1)
            else:
                action = np.argmax(Q_value[state_current, :])

            state_next = action
            ti = time_array[state_next]
            err = error_array[state_next]

            if ti < min(arr_min_time):
                ti_reward += 1
            elif ti == min(arr_min_time):
                ti_reward += 0.1
            else:
                ti_reward += 0

            if err < min(arr_min_error):
                err_reward += 1
            elif err == min(arr_min_error):
                err_reward += 0.1
            else:
                err_reward += 0

            si_current = 100 * err + 10 * ti
            add_reward = 100 * err_reward + 10 * ti_reward
            R[state_current, action] = R[state_current, action] + add_reward
            TD = R[state_current, action] + gamma * max(Q_value[state_next, :]) - Q_value[state_current, action]
            Q_value[state_current, action] = Q_value[state_current, action] + alpha * TD

            state_current = state_next
            pre_ti = ti
            pre_err = err

            if si_current <= min(arr_si):
                break

    print("Q Learning Calculated")
    print(Q_value)
    max_value = [max(Q_value[:, n]) for n in range(0, size)]
    cov_rate = np.argmax(max_value) + 1
    print(max(max_value))
    print(cov_rate)

    Q_plt = Q_value.reshape(-1)
    action_arr = []
    state_arr = []
    action_arr_plt = []
    state_error_arr_plt = []
    state_time_arr_plt = []

    for n in range(0, size):
        for m in range(0, size):
            action_arr_plt.append(m + 1)
            state_error_arr_plt.append(error_array[n])
            state_time_arr_plt.append(time_array[n])

    for k in range(0, 22500):
        with open("Qlearn_data_1.txt", "a") as f:
            f.write(str(action_arr_plt[k]) + "\t" + str(state_error_arr_plt[k]) + "\t" + str(Q_plt[k]) + "\n")
        with open("Qlearn_data_2.txt", "a") as f:
            f.write(str(action_arr_plt[k]) + "\t" + str(state_time_arr_plt[k]) + "\t" + str(Q_plt[k]) + "\n")

    # cmap = plt.cm.get_cmap('jet') # Get desired colormap - you can change this!
    # max_height = np.max(Q_plt)   # get range of colorbars so we can normalize
    # min_height = np.min(Q_plt)
    # # scale each z to [0,1], and get their rgb values
    # rgba = [cmap((k-min_height)/max_height) for k in Q_plt] 
    
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111, projection='3d')
    # x = state_arr_plt
    # y = action_arr_plt
    # z = np.zeros(22500)
    # dx = 5*np.ones(22500)
    # dy = 5*np.ones(22500)
    # dz = Q_plt
    # ax1.set_title("Q table")
    # ax1.bar3d(x, y, z, dx, dy, dz,shade=True,color=rgba)
    # ax1.set_xlabel("State")
    # ax1.set_ylabel("Action")
    # ax1.set_zlabel("Q value")
    # plt.show()

