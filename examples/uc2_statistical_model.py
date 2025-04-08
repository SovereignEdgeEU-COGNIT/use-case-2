import asyncio
import logging
import math
import os
import random
import sys
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

import utils
from cognit import device_runtime
from uc2_function import fire_presence_detection

# silence cognit logger
cognit_logger = logging.getLogger("cognit-logger")
cognit_logger.setLevel(100)

# silence warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("uc2")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

plt.ion()

DATA_DIR = os.path.join(os.path.dirname(__file__),'data')

async def offload_function(image_path: str, x: int, y: int) -> tuple[str | bool, int, int]:
    """Offloads the image recognition function given an image path and coordinates.

    Args:
        image_path (str): The path to the image to be processed.
        x (int): The x coordinate of the point.
        y (int): The y coordinate of the point.

    Returns:
        tuple[str | bool, int, int]: A tuple consisting of:
            - The result of the image recognition function.
                A string with the error if one occurred, a bool otherwise indicating the result of the detection.
            - The x coordinate of the point.
            - The y coordinate of the point.
    """
    
    REQS_INIT = {
          "FLAVOUR": "UC2_V2",
          "MIN_ENERGY_RENEWABLE_USAGE": 85,
          "GEOLOCATION": "IKERLAN ARRASATE/MONDRAGON 20500"
    }
    
    image = cv2.imread(image_path)
    image_lst = image.tolist()
    try:
        # Instantiate a device Device Runtime
        my_device_runtime = device_runtime.DeviceRuntime("./cognit-uc2.yml")
        my_device_runtime.init(REQS_INIT)
        # Offload and execute a function
        return_code, result = await asyncio.to_thread(my_device_runtime.call, fire_presence_detection,image_lst)
        logger.debug('Status code: %d', return_code)
        
        logger.info('x: %d, y: %d, image: %s, result: %s', x, y, image_path, result)
        
        return result, x, y
        
    except Exception as e:
        logger.error("An exception has occured: %s", str(e))
        exit(-1)

async def activate_sensor(row: int, col: int, prob: float):
    """Activate sensor at given coordinates, as well as nearby devices, based on fire probability.

    Args:
        row (int): The x coordinate of the point.
        col (int): The y coordinate of the point.
        prob (float): The base probability of fire detection.
    """
    global matrix_center,triggered_devices,img,fig

    offloaded_functions = []
   
    for i in range(row - wake_up_distance, row + wake_up_distance + 1):
        for j in range(col - wake_up_distance, col + wake_up_distance + 1):
            
            trigger = max(random.random(), 0.01)
            dist_x = abs(matrix_center - i)
            dist_y = abs(matrix_center - j)
            
            if(0 <= i < len(matrix) and 0 <= j < len(matrix)):
                if (matrix[i][j] == 0):
                    if((i != matrix_center or j != matrix_center) and (trigger < utils.detection_probability(dist_x, dist_y, prob))):
                        offloaded_functions.append(offload_function(os.path.join(DATA_DIR, "fire2.png"),i,j))
                    else:
                        offloaded_functions.append(offload_function(os.path.join(DATA_DIR, "wood2.png"),i,j))
                else:
                    # TODO: Coordinates were row, col should it be i,j?
                    offloaded_functions.append(offload_function(os.path.join(DATA_DIR, "fire2.png"),row,col))
    logger.debug('Offloaded %d functions', len(offloaded_functions))
    results = await asyncio.gather(*offloaded_functions)
    for result in results:
        v, i, j = result
        logger.debug(v,i,j)
        logger.debug('Result: %s', v)
        if isinstance(v, str):
            logger.error('Error encountered when offloading function for i: %d j: %d:\n%s', i, j, v)
        if (v) and (matrix[i][j] == 0):
            matrix[i][j] = 1
            triggered_devices += 1
            if (triggered_devices == number_of_devices):
                logger.info('DONE')
                sys.exit()
            img.set_data(matrix)
            fig.canvas.flush_events()
            await activate_sensor(i,j,prob)

number_of_devices = 115
wake_up_distance = 1
triggered_devices = 1

col = 0
row = 0
matrix_size = math.ceil(np.sqrt(number_of_devices))
number_of_cycles = 0

device_per_area = 1
probability = 0.10
probability_increment = 0.05


if (matrix_size%2 == 0):
    matrix_size += 1

matrix_center = matrix_size//2

matrix = np.zeros((matrix_size, matrix_size), dtype=np.bool_)

matrix[matrix_center, matrix_center] = 1

fig = plt.figure()
img = plt.imshow(matrix)
fig.canvas.flush_events()

while triggered_devices < number_of_devices:
    
    logger.info("-----------------------Cycle %d----------------------", number_of_cycles)
    logger.info("Fire probability %.2f", probability)

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val == 1:
                asyncio.run(activate_sensor(i,j,probability))
    
    probability += probability_increment
    number_of_cycles += 1