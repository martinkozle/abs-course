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
    actions = ('E', 'W', 'N', 'S')
    Q = {
        (state, action): 0 for state in states for action in (*actions, 'exit')
    }
    delta = 0.5
    alpha = 0.5
    episodes = [
        (
            ((1, 3), 'S', (1, 2), 0),
            ((1, 2), 'E', (2, 2), 0),
            ((2, 2), 'S', (2, 1), -100)
        ),
        (
            ((1, 3), 'S', (1, 2), 0),
            ((1, 2), 'E', (2, 2), 0),
            ((2, 2), 'E', (3, 2), 0),
            ((3, 2), 'N', (3, 3), 100)
        ),
        (
            ((1, 2), 'S', (1, 2), 0),
            ((1, 2), 'E', (2, 2), 0),
            ((2, 2), 'E', (3, 2), 0),
            ((3, 2), 'S', (3, 1), 80)
        )
    ]
    for i, episode in enumerate(episodes):
        print(f'Episode {i+1}')
        Q_new = Q.copy()  # NOSONAR
        for state, action, new_state, reward in episode:
            m = max(Q[(new_state, action)] for action in actions)
            Q_new[(state, action)] = (
                (1 - delta) * Q[(state, action)]
                + delta * (
                    reward
                    + alpha * m
                )
            )
            print(
                f'Qnew({state}, {action}) = 0.5 * Q({state}, {action})'
                f' + 0.5 * ({reward} + 0.5 * {m})'
                f' = {Q_new[state, action]}'
            )
        Q = Q_new
        # print(Q)
    print(f'Q((3, 2), N) = {Q[(3, 2), "N"]}')
    print(f'Q((1, 2), S) = {Q[(1, 2), "S"]}')
    print(f'Q((2, 2), E) = {Q[(2, 2), "E"]}')


if __name__ == '__main__':
    main()
