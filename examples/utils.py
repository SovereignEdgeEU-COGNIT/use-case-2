def detection_probability(dist_x: int, dist_y: int, base_probability: float) -> float:
    """Calculates detection probability based on distance from the center.

    Args:
        dist_x (int): Distance in the x direction from the center
        dist_y (int): Distance in the y direction from the center
        base_probability (float): Base probability of detection

    Returns:
        float: Calculated detection probability
    """
    return 0.5**(max(abs(dist_x),abs(dist_y)))*base_probability