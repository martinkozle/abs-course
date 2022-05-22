def make_move(state, action, moves):
    new_state = (state[0] + moves[action][0], state[1] + moves[action][1])
    if (
        new_state[0] <= 0 or new_state[0] > 3
        or new_state[1] <= 0 or new_state[1] > 3
    ):
        return state
    return new_state


def main():
    states = (
        (1, 2), (1, 3), (2, 2), (3, 2), (1, 1), (2, 1), (2, 3), (3, 1), (3, 3)
    )
    exit_states = ((1, 1), (2, 1), (2, 3), (3, 1), (3, 3))
    actions = ('E', 'W', 'N', 'S')
    Q = {
        (state, action): 0 for state in states for action in (*actions, 'exit')
    }
    R = {(1, 1): 25, (2, 1): -100, (2, 3): -80, (3, 1): 80, (3, 3): 100}
    moves = {'E': (1, 0), 'W': (-1, 0), 'N': (0, 1), 'S': (0, -1)}
    delta = 0.5
    alpha = 0.5
    for k in range(3):
        print(f'k={k+1}')
        Q_new = Q.copy()  # NOSONAR
        for state in states:
            if state in exit_states:
                Q_new[state, 'exit'] = (
                    (1 - delta) * Q[state, 'exit']
                    + delta * R.get(state, 0)
                )
                print(
                    f'Qnew({state}, exit) = 0.5 * Q({state}, exit) '
                    f'+ 0.5 * R({state})'
                    f' = {Q_new[state, "exit"]}'
                )
                continue
            for action in actions:
                new_state = make_move(state, action, moves)
                if new_state in exit_states:
                    m = Q[(new_state), 'exit']
                    s = f'Q({new_state}, exit)'
                else:
                    m = max(Q[(new_state, action)] for action in actions)
                    s = f"max(Q({new_state}, a')))"
                Q_new[state, action] = (
                    (1 - delta) * Q[state, action]
                    + delta * (R.get(state, 0) + alpha * m)
                )
                print(
                    f'Qnew({state}, {action}) = 0.5 * Q({state}, {action}) '
                    f'+ 0.5 * (R({state}) + 0.5 * {s}'
                    f' = {Q_new[state, action]}'
                )
                if action == 'exit':
                    break
        Q = Q_new


if __name__ == '__main__':
    main()
