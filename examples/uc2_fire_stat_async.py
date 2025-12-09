from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
from bokeh_server import BokehServerStat
import uc2_utils

logger = logging.getLogger('uc2')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

DATA_DIR = os.path.join(os.path.dirname(__file__),'data')
FIRE_IMG_PATH = os.path.join(DATA_DIR, 'fire2.png')
WOOD_IMG_PATH = os.path.join(DATA_DIR, 'wood2.png')

class StatAsync:
    def __init__(self, n_devices: int, wakeup_distance: int = 1, devices_per_area: int = 1, probability: float = 0.1, probability_increment: float = 0.05, plotting: bool = False) -> None:
        self.n_devices = n_devices
        self.wakeup_distance = wakeup_distance
        self.devices_per_area = devices_per_area
        
        self.probability = probability
        self.probability_increment = probability_increment
        
        # Create a matrix to represent the device areas
        self._matrix_side = np.ceil(np.sqrt(n_devices)).astype(int)
        if (self._matrix_side % 2 == 0):
            self._matrix_side += 1
        self._matrix_center = self._matrix_side // 2
        self.matrix = np.zeros((self._matrix_side, self._matrix_side), dtype=np.bool_)
        
        self.plotting = plotting
        
        if self.plotting:
            # Set up Bokeh server for visualization
            self.bokeh_server = BokehServerStat(self.matrix)
            self.bokeh_server.update_requested = False
            self.bokeh_server.start_bokeh_server()
        else:
            self.bokeh_server = None

        # Initialize cycle
        self.reset()

    def reset(self):
        # Reset matrix
        self.matrix.fill(0)
        # Activate the center device
        self.matrix[self._matrix_center, self._matrix_center] = 1
        self.update_map()
        # Reset triggered devices & cycles
        self._triggered_devices = 1
        self._number_of_cycles = 0
        
        self.running = False

    def update_map(self):
        if self.plotting:
            self.bokeh_server.matrix = self.matrix
            self.bokeh_server.update_requested = True 
    
    def cleanup(self):
        if self.plotting and self.server:
            # Stop the Bokeh server
            self.server.stop()


    def get_activated_devices(self) -> list[uc2_utils.Coordinates]:
        activated_deices = []
        for x, y in zip(*np.where(self.matrix == 1)):
            activated_deices.append(uc2_utils.Coordinates(x,y))
        return activated_deices

    async def send_requests(self, active_devices: list[uc2_utils.Coordinates], edge_devices: list[uc2_utils.Coordinates]):
        offloaded_functions = []
        for active_device in active_devices:
            for _ in range(self.devices_per_area):
                offloaded_functions.append(uc2_utils.offload_function(FIRE_IMG_PATH, active_device))
            
        for edge_device in edge_devices:
            trigger = max(np.random.random(), 0.01)
            dist_x = abs(self._matrix_center - edge_device.x)
            dist_y = abs(self._matrix_center - edge_device.y)
            if (trigger < uc2_utils.detection_probability(dist_x, dist_y, self.probability)):
                for _ in range(self.devices_per_area):
                    offloaded_functions.append(uc2_utils.offload_function(FIRE_IMG_PATH, edge_device))
            else:
                for _ in range(self.devices_per_area):
                    offloaded_functions.append(uc2_utils.offload_function(WOOD_IMG_PATH, edge_device))
            
        logger.debug('Offloaded %d functions', len(offloaded_functions))
        logger.debug('There are %d edge devices', len(edge_devices))
        results = await asyncio.gather(*offloaded_functions)
        for result in results:
            v, loc = result
            logger.debug('Result: %s for %i %i', v, loc.x, loc.y)
            if isinstance(v, str):
                logger.error('Error encountered when offloading function for i: %d j: %d:\n%s', loc.x, loc.y, v)
                continue
            
            self.matrix[loc.x][loc.y] |= v
            self._triggered_devices = np.sum(self.matrix)
            self.update_map()
            if self._triggered_devices == self.n_devices:
                self.running = False
                logger.info('DONE')
                return
    async def cycle(self):
        while (self._triggered_devices < self.n_devices) and self.running:
            logger.info("-----------------------Cycle %d----------------------", self._number_of_cycles)
            logger.info("Fire probability %.2f", self.probability)
            
            activated_devices = self.get_activated_devices()
            edge_devices = uc2_utils.find_nearby_from_list(activated_devices, self.wakeup_distance, self._matrix_side)
            logger.debug("Activated devices: %s", activated_devices)
            logger.debug("Edge devices: %s", edge_devices)
            # Send requests to offload functions
            await self.send_requests(activated_devices, edge_devices)
            
            # Increment probability and cycle count
            self.probability += self.probability_increment
            self._number_of_cycles += 1

    def run(self):
        self.running = True
        asyncio.run(self.cycle())

            
            
if __name__ == "__main__":

    logger.setLevel(logging.INFO)
    
    number_of_devices = 115    
    stat_async = StatAsync(number_of_devices, plotting=True)
    
    try:
        stat_async.run()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")