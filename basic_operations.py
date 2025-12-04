import math 

def mean(data):
    """
    Replaces np.mean(data)
    Formula: Σx / N
    """
    if len(data) == 0:
        return 0.0
    return sum(data) / len(data)

def var(data):
    """
    Replaces np.var(data)
    Formula: Σ(x - mean)^2 / N
    Note: This calculates Population Variance (divides by N), 
    which is the default behavior of np.var.
    """
    if len(data) == 0:
        return 0.0
    
    mu = mean(data)
    
    # Calculate sum of squared differences
    squared_diffs = sum([(x - mu)**2 for x in data])
    
    return squared_diffs / len(data)

def std(data):
    """
    Replaces np.std(data)
    Formula: sqrt(variance)
    """
    variance = var(data)
    return math.sqrt(variance)


