from math import ceil
from typing import List, Dict, Set

# TODO this is all kindof worthless since much of the time there is no value of
# n that produces the desired size and one has to approximate. Doing it manually is enough.
def calc_n_for_1_of_n_examples(total_items: int, desired_items: int) -> int:
    """
    For a given number of total items and a desired size of a subset of those items,
    calculate a value, n, for a sample_1_of_n_examples setting that will lead to a subset
    of the desired size.
    """

    # TODO Ideally we'd allow values of 0 I think, but idk the behavior downstream if the dataset is empty
    if not total_items >= 1: raise ValueError("There must be at least 1 data item")
    if not desired_items >= 1: raise ValueError("The resulting number of samples must be at least 1")

    if desired_items == 1: return total_items

    # Effect of n is based on Dataset.shard(), which works out to:
    # ceil(total_items / n) => desired_items
    # So, looking for integer n such that:
    # total_items / desired_items < n <= total_items / (desired_items - 1)
    n: int = ceil(total_items/desired_items)

    # There may not be an integer between those two values
    if n >= total_items/(desired_items-1):
        raise ValueError(f"There is no valid shard-based sampling which results in {desired_items} samples from {total_items} total items")
    return n


def calc_n_for_1_of_n_examples_test():

    def n_to_desired_size(total_items: int, n: int) -> int:
        """
        Simulate how many items Dataset.shard will return, given the total num of items 
        and the configured value n for sample_1_of_n_examples
        """
        return len([i for i in range(total_items) if i % n == 0])

    # I can finally write Python that looks like Java :S
    results: List[Dict[int, Set[int]]] = []

    # Go through all the total_size/n combos and record what desired_size vals they create
    largest_total_size=100
    for total_items in range(1, largest_total_size):
        local_results: Dict[int, Set[int]] = {}
        for n in range(1, total_items + 1):
            desired_size: int = n_to_desired_size(total_items, n)
            if not desired_size in local_results:
                local_results[desired_size] = set()
            local_results[desired_size].add(n)
        results += [local_results]

    # Compare results of calc_n_for_1_of_n_examples with simulation above to see if the fn
    # correctly predicts a value of n that will return desired size.
    for idx, desired_size_and_n in enumerate(results):
        total_items = idx + 1
        for desired_size in range(1, total_items + 1):
            if desired_size in desired_size_and_n:
                n = calc_n_for_1_of_n_examples(total_items, desired_size)
                assert n in desired_size_and_n[desired_size], \
                    f"n:{n} is not an expected value for n:{desired_size_and_n[desired_size]}" \
                    f", total_items:{total_items}, desired_size:{desired_size}"
            else:
                try:
                    # Some desired_sizes are impossible and should throw errors
                    calc_n_for_1_of_n_examples(total_items, desired_size)
                    assert False, f"Expected exception for desired_size:{desired_size} out of total_items:{total_items}"
                except ValueError:
                    pass

calc_n_for_1_of_n_examples_test()