import numpy as np

def derangement_indices(n: int, seed: int | None = None) -> np.ndarray:
    """
    Возвращает перестановку 0..n-1 без неподвижных точек (дерранжировку).
    Реализация — алгоритм Саттоло (один цикл длины n).
    """
    if n < 2:
        raise ValueError("Дерранжировка не существует для n < 2.")
    rng = np.random.default_rng(seed)
    a = np.arange(n)
    # Sattolo: на каждом шаге выбираем j < i, тем самым j != i
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i)  # 0 <= j < i
        a[i], a[j] = a[j], a[i]
    return a

def reshuffle_first_axis(data: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (перемешанный_data, новые_индексы),
    где перемешивание по первому измерению гарантирует,
    что ни один элемент не остаётся на месте.
    """
    order = derangement_indices(data.shape[0], seed=seed)
    return data[order], order

def main():
    n_ch = 16
    order = derangement_indices(n_ch)
    from pprint import pprint
    pprint(order)
    # print('[',(*order, sep=', '),']')

if __name__ == '__main__':
    main()