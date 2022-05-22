from collections import defaultdict


def main():
    Q = defaultdict(int)
    episode = (
        ((1, 3), 'S', (1, 2), 0),
        ((1, 2), 'E', (2, 2), 0),
        ((2, 2), 'S', (2, 1), -100)
    )
    f3 = {'N': 1, 'S': 2, 'E': 3, 'W': 4}
    w1 = 0
    w2 = 0
    w3 = 0
    alpha = 0.5
    for state, action, new_state, reward in episode:
        Q_new = Q.copy()  # NOSONAR
        Q_new[state, action] = w1 * state[0] + w2 * state[1] + w3 * f3[action]
        difference = (
            reward + max(Q[new_state, action] for action in 'NESW')
            - Q_new[state, action]
        )
        w1 += alpha * difference * state[0]
        w2 += alpha * difference * state[1]
        w3 += alpha * difference * f3[action]
        print(
            f'Qnew({state}, {action}) = {w1} * {state[0]} + {w2} * {state[1]}'
            f' + {w3} * {f3[action]} = {Q_new[state, action]}'
        )
        print(
            f"difference = reward + max(Q({state}, a')"
            f' - Q({state}, {action})'
            f' = {difference}'
        )
        print(f'w1 = alpha * difference * {state[0]} = {w1}')
        print(f'w2 = alpha * difference * {state[1]} = {w2}')
        print(f'w3 = alpha * difference * {f3[action]} = {w3}')
        Q = Q_new

    for action in 'NESW':
        q = 1 * 2 + 1 * 2 + 1 * f3[action]
        print(
            f'Qnew({state}, {action}) = {1} * {2} + {1} * {2}'
            f' + {1} * {f3[action]} = {q}'
        )


if __name__ == '__main__':
    main()
