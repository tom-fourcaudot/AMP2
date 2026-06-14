from statistics import stdev, mean


def get_stats(accuracies: list[float]) -> tuple[float, float]:
    avg = mean(accuracies)
    variance = stdev(accuracies)
    return avg, variance