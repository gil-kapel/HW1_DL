def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    current_state = goal_state
    next_action = None
    while current_state.to_string() in prev.keys():
        result.append((current_state, next_action))
        if prev[current_state.to_string()] is None:
            break
        current_state, next_action = prev[current_state.to_string()]
    result.reverse()
    return result[:-1]


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    final_plan = []
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
            final_plan.append(action)
    # print(''.join(final_plan))
