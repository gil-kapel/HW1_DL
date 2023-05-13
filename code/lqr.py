import numpy as np
from cartpole_cont import CartPoleContEnv
import matplotlib.pyplot as plt


def get_A(cart_pole_env):
    '''
    create and returns the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the A matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    g = cart_pole_env.gravity
    pole_mass = cart_pole_env.masspole  #m
    cart_mass = cart_pole_env.masscart #M
    pole_length = cart_pole_env.length #l
    dt = cart_pole_env.tau
    return np.matrix(np.eye(4) + np.multiply(dt, [[0, 1, 0, 0],
                                                [0, 0, (pole_mass*g/cart_mass), 0],
                                                [0, 0, 0, 1],
                                                [0, 0, (g/pole_length)*(1+(pole_mass/cart_mass)), 0]]))


def get_B(cart_pole_env):
    '''
    create and returns the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    :param cart_pole_env: to extract all the relevant constants
    :return: the B matrix used in LQR. i.e. x_{t+1} = A * x_t + B * u_t
    '''
    cart_mass = cart_pole_env.masscart
    pole_length = cart_pole_env.length
    dt = cart_pole_env.tau

    return np.matrix(np.multiply(dt, [[0],
                                      [1/cart_mass],
                                      [0],
                                      [1/(cart_mass*pole_length)]]))


def find_lqr_control_input(cart_pole_env):
    """
    implements the LQR algorithm
    :param cart_pole_env: to extract all the relevant constants
    :return: a tuple (xs, us, Ks). xs - a list of (predicted) states, each element is a numpy array of shape (4,1).
    us - a list of (predicted) controls, each element is a numpy array of shape (1,1). Ks - a list of control transforms
    to map from state to action, np.matrix of shape (1,4).
    """
    assert isinstance(cart_pole_env, CartPoleContEnv)
    cart_pole_env.reset()

    A = get_A(cart_pole_env)
    B = get_B(cart_pole_env)

    w1 = 0.5
    w2 = 1.0
    w3 = 0.5

    Q = np.matrix([[w1, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, w2, 0],
                   [0, 0, 0, 0]])
    R = np.matrix([w3])

    T = cart_pole_env.planning_steps
    Ps, us = [Q], []
    xs = [np.expand_dims(cart_pole_env.state, 1)]

    for _ in range(T):
        Pt = Q + A.T @ Ps[0] @ A - A.T @ Ps[0] @ B @ np.linalg.inv(B.T @ Ps[0] @ B + R) @ B.T @ Ps[0] @ A
        Ps = [Pt] + Ps

    Ks = [-np.linalg.inv(B.T @ Ps[t+1] @ B + R) @ B.T * Ps[t+1] @ A for t in range(T)]

    for t in range(T):
        ut = Ks[t] @ xs[t]
        next_xs, _, is_done, _ = env.step(np.squeeze(np.asarray(ut), 1))
        us.append(ut)
        xs.append(np.expand_dims(next_xs, 1))
        if is_done:
            break

    assert len(xs) == cart_pole_env.planning_steps + 1, "if you plan for x states there should be X+1 states here"
    assert len(us) == cart_pole_env.planning_steps, "if you plan for x states there should be X actions here"
    for x in xs:
        assert x.shape == (4, 1), "make sure the state dimension is correct: should be (4,1)"
    for u in us:
        assert u.shape == (1, 1), "make sure the action dimension is correct: should be (1,1)"
    return xs, us, Ks


def print_diff(iteration, planned_theta, actual_theta, planned_action, actual_action):
    print('iteration {}'.format(iteration))
    print('planned theta: {}, actual theta: {}, difference: {}'.format(
        planned_theta, actual_theta, np.abs(planned_theta - actual_theta)
    ))
    print('planned action: {}, actual action: {}, difference: {}'.format(
        planned_action, actual_action, np.abs(planned_action - actual_action)
    ))


if __name__ == '__main__':
    final_theta = []
    verbose = False
    render = True
    q2_params = (np.linspace(-0.1, 0.1, 10) * np.pi, True)
    unstable = 0.364
    q3_params = (np.array([0.1, unstable, 0.5 * unstable]) * np.pi, True)

    unstable = 0.00003
    q4_params = (np.array([0.1, unstable, 0.5 * unstable]) * np.pi, False)

    unstable = 0.22
    q5_params = (np.array([0.1, unstable, 0.5 * unstable]) * np.pi, True, 4.0)

    chosen_params = q4_params

    if len(chosen_params) <= 2:
        env = CartPoleContEnv(initial_theta=0.1)
    else:
        env = CartPoleContEnv(initial_theta=0.1, force_limit=chosen_params[2])

    # # print the matrices used in LQR
    print('A: {}'.format(get_A(env)))
    print('B: {}'.format(get_B(env)))
    # use LQR to plan controls
    xs, us, Ks = find_lqr_control_input(env)

    for theta in chosen_params[0]:
        if len(chosen_params) <= 2:
            env = CartPoleContEnv(initial_theta=theta)
        else:
            env = CartPoleContEnv(initial_theta=theta, force_limit=chosen_params[2])
        actual_state = env.reset()
        # run the episode until termination, and print the difference between planned and actual
        is_done = False
        iteration = 0
        is_stable_all = []
        thetas = []
        while not is_done:
            if render:
                env.render()
            # print the differences between planning and execution time
            predicted_theta = xs[iteration].item(2)
            actual_theta = actual_state[2]
            thetas.append((actual_theta + np.pi) % (2 * np.pi) - np.pi)
            predicted_action = us[iteration].item(0)
            if chosen_params[1]:
                actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1)).item(0)
            else:
                actual_action = us[iteration][0, 0]
            if verbose:
                print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
            # apply action according to actual state visited
            # make action in range
            actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
            actual_action = np.array([actual_action])
            actual_state, reward, is_done, _ = env.step(actual_action)
            is_stable = reward == 1.0
            is_stable_all.append(is_stable)
            iteration += 1
        env.close()
        final_theta.append(thetas)
    # we assume a valid episode is an episode where the agent managed to stabilize the pole for the last 100 time-steps
        valid_episode = np.all(is_stable_all[-100:])
        # print if LQR succeeded
        print('valid episode: {}'.format(valid_episode))

    # plot graph
    final_theta = np.array(final_theta).T
    fig, ax = plt.subplots()
    ax.plot(final_theta)
    ax.legend([f"{s/np.pi:.2}*pi" for s in chosen_params[0]])
    plt.title("Theta over time")
    plt.xlabel("time step")
    plt.ylabel("theta [radians]")
    # fig.savefig("LQR_theta_over_time.png")
    plt.show()
