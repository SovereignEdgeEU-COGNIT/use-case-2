from __future__ import annotations

import asyncio
import logging
import os
from typing import NamedTuple, TypeVar

import cv2
import numpy as np
from cognit import device_runtime
from shapely.affinity import rotate, scale
from shapely.geometry import Point, Polygon

from uc2_function import fire_presence_detection

logger = logging.getLogger('uc2')

class Coordinates(NamedTuple):
    x: int
    y: int

    def __add__(self, other: Coordinates) -> Coordinates:
        return Coordinates(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Coordinates) -> Coordinates:
        return Coordinates(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Coordinates:
        return Coordinates(int(self.x * scalar), int(self.y * scalar))
    
    def __eq__(self, other: Coordinates) -> bool:
        return self.x == other.x and self.y == other.y


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

def find_nearby(position: Coordinates, distance_from_pos: int, side_length: int) -> set[Coordinates]:
    """Finds nearby coordinates within the distance_from_pos distance."""
    nearby_coords = []
    for dx in range(-distance_from_pos, distance_from_pos + 1):
        for dy in range(-distance_from_pos, distance_from_pos + 1):
            new_coord = Coordinates(position.x + dx, position.y + dy)
            if (0 <= new_coord.x < side_length and
                    0 <= new_coord.y < side_length):
                nearby_coords.append(new_coord)
    return set(nearby_coords) - {position}

def find_nearby_from_list(positions: list[Coordinates], distance_from_pos: int, side_length: int) -> set[Coordinates]:
    """Finds nearby coordinates within the distance_from_pos distance from a list of positions."""
    nearby_coords = []
    for pos in positions:
        nearby_coords.extend(find_nearby(pos, distance_from_pos, side_length))
    return set(nearby_coords) - set(positions)

def generate_random_points_in_polygon(polygon: Polygon, num_points:int = 100) -> list[Point]:
    
    minx, miny, maxx, maxy = polygon.bounds  # Get bounding box of polygon
    points = []
    
    while len(points) < num_points:
        # Generate random points within the bounding box
        random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        
        # Check if the point is within the polygon
        if polygon.contains(random_point):
            points.append(random_point)
            
    return points

def generate_evently_spaced_points_in_polygon(polygon: Polygon, spacing: float) -> list[Point]:
    minx, miny, maxx, maxy = polygon.bounds  # Get bounding box of polygon
    points = []
    coordinates = np.mgrid[-minx:maxx:spacing, miny:maxy:spacing].reshape(2,-1).T
    for x, y in coordinates:
        point = Point(x, y)
        if polygon.contains(point):
            points.append(point)
            
    return points

def elliptical_buffer(center_point: Point, major_axis: float, minor_axis: float, angle: float = 0., resolution:int = 36) -> Polygon:
    """
    Creates an elliptical buffer around a point.

    Args:
        point (Point): The center point of the ellipse.
        major_axis (float): The length of the major axis.
        minor_axis (float): The length of the minor axis.
        angle (float): The rotation angle of the ellipse (in degrees).
        resolution (int): The number of points used to create the circle.

    Returns:
        Polygon: The elliptical buffer polygon.
    """

    # Create a circular buffer
    circle = center_point.buffer(1, resolution=resolution)

    # Scale the circle to create an ellipse
    ellipse = scale(circle, xfact=major_axis, yfact=minor_axis)

    # Rotate the ellipse
    rotated_ellipse = rotate(ellipse, angle, origin=center_point)

    return rotated_ellipse

def move_point(point: Point, distance: float, direction: float) -> Point:
    """Moves a point in a given direction by a specified distance.

    Args:
        point (Point): The original point.
        distance (float): The distance to move by.
        direction (float): The direction in degrees.

    Returns:
        Point: The new point after moving.
    """
    # Convert direction from degrees to radians
    direction_rad = np.deg2rad(direction)
    
    # Calculate the new coordinates
    new_x = point.x + distance * np.cos(direction_rad)
    new_y = point.y + distance * np.sin(direction_rad)
    
    return Point(new_x, new_y)

T = TypeVar('T')
async def offload_function(image_path: str, loc: T) -> tuple[str | bool, T]:
    """Offloads the image recognition function given an image path and coordinates.

    Args:
        image_path (str): The path to the image to be processed.
        loc (T): The coordinates of the device. Just used to return the coordinates

    Returns:
        tuple[str | bool, int, int]: A tuple consisting of:
            - The result of the image recognition function.
                A string with the error if one occurred, a bool otherwise indicating the result of the detection.
            - The coordinates of the device.
    """
    REQS_INIT = {
    "ID": "deviceNatureFR1",
    "FLAVOUR": "NatureFR",
    # "PROVIDERS": ["Nature4"],
    "IS_CONFIDENTIAL": False,
    "GEOLOCATION": {
        "latitude": 42.449,
        "longitude": 12.0864,
    },
}
    if not isinstance(image_path, str):
        image_path = str(image_path)
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
    image_list = resized_image.tolist()
    try:
        # Instantiate a device Device Runtime
        my_device_runtime = device_runtime.DeviceRuntime("./cognit-uc2.yml")
        my_device_runtime.init(REQS_INIT)
        # Offload and execute a function
        ret = await asyncio.to_thread(my_device_runtime.call, fire_presence_detection,image_list)
        my_device_runtime.stop()
        return_code = ret.ret_code
        result = ret.res
        _ = ret.err
        logger.debug('Status code: %s', return_code)
        
        logger.info('loc: %s, image: %s, result: %s', loc, image_path, result)
        
        return result, loc
        
    except Exception as e:
        logger.error("An exception has occured: %s", str(e))
        # exit(-1)
        return str(e), loc


if __name__ == '__main__':
    # Small script to test cognit
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fire_img = os.path.join(file_dir, 'data','fire2.png')
    wood_img = os.path.join(file_dir, 'data','wood2.png')
    print('Offloading function with fire')
    r, _ = asyncio.run(offload_function(fire_img, None))
    print(r)
    print('Offloading function with fire')
    r, _ = asyncio.run(offload_function(wood_img, None))
    print(r)
    