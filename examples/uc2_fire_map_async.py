import asyncio
import logging
import os
import warnings
import webbrowser
from pathlib import Path
from time import sleep

import fiona
import geopandas as gpd
import tomli
from bokeh.models import GMapOptions
from dotenv import load_dotenv
from shapely.geometry import Polygon

import uc2_utils
from bokeh_server import BokehServerMap

logging.root.setLevel(logging.INFO)
logger = logging.getLogger('uc2')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore')

class MapAsync:
    def __init__(self, settings_file: str | None = None, settings: dict | None = None, plotting: bool = False) -> None:

        # Load settings
        if settings_file is not None:
            self._init_from_file(settings_file)
        elif settings is not None:
            self._init_from_dict(settings)
        else:
            raise ValueError('Either settings_file or settings must be provided.')

        # Load API key
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if self.api_key is None:
            raise ValueError('GOOGLE_MAPS_API_KEY environment variable not set.')
        
        os.makedirs(self.maps_save_path, exist_ok=True)
        
        self._configure_map_data()
        
        self.plotting = plotting
        if self.plotting:
            self._map_idx = 0
            self.browser = webbrowser.get()
            self._create_plot()

    @property
    def fire_speed(self) -> float:
        return self.fire_speed_multiplier*self.wind_speed*0.0001

    @property
    def park_filepath(self) -> Path:
        return self.park_dir / self.park_file

    def _init_from_file(self, file):
        if not os.path.exists(file):
            raise FileNotFoundError(f'Settings file {file} does not exist.')
        with open(file, 'rb') as f:
            settings = tomli.load(f)
        self._init_from_dict(settings)

    def _init_from_dict(self, settings: dict):
        map_settings = settings.get('map', {})
        self.wind_speed = map_settings.get('wind_speed', 100)
        self.fire_speed_multiplier = map_settings.get('fire_speed_multiplier', 0.1)
        self.wind_direction = map_settings.get('wind_direction', 0)
        self.maps_save_path = Path(map_settings.get('map_base_path', 'saved_maps'))
        self.park_dir = Path(map_settings.get('park_dir', 'maps'))
        self.park_file = Path(map_settings.get('park_file', 'park.kml'))
        self.initial_fire_size = map_settings.get('initial_fire_size', 0.1)
        self.smoke_area_offset = map_settings.get('smoke_area_offset', 10)
        
        general_settings = settings.get('general', {})
        self.n_devices = general_settings.get('number_of_devices')
        self.device_spacing = general_settings.get('device_spacing')
        if self.n_devices is None and self.device_spacing is None:
            self.n_devices = 50  # Defaults to 50 devices
        self.device_trigger_radius = general_settings.get('device_trigger_radius', 1)
        self.simulations_steps = general_settings.get('simulation_steps', 10)
        
        path_settings = settings.get('common-paths', {})
        self.data_path = Path(path_settings.get('data_path', 'data'))
        self.fire_img_file = self.data_path / Path(path_settings.get('fire_img_filename', 'fire.png'))
        self.wood_img_file = self.data_path / Path(path_settings.get('wood_img_filename', 'wood.png'))
        
    def _update_settings_from_dashboard(self):
        if self.bokeh_server.is_initialized():
            park_name = self.bokeh_server.map_name_select.value
            updated = False
            if self.park_file != f'{park_name}.kml':
                updated = True
                self.park_file = f'{park_name}.kml'
            if self.wind_direction != self.bokeh_server.wind_direction_slider.value:
                updated = True
                self.wind_direction = self.bokeh_server.wind_direction_slider.value
            if self.wind_speed != self.bokeh_server.wind_speed_slider.value:
                updated = True
                self.wind_speed = self.bokeh_server.wind_speed_slider.value
            if updated:
                self._configure_map_data()
                map_options = GMapOptions(lat=self.park_polygon.centroid.y, lng=self.park_polygon.centroid.x, map_type="terrain", zoom=10)
                self.bokeh_server.update_map_data(map_options=map_options, park=self.park_polygon, fire_point=self.fire_point, device_data=self.devices_frame)

    def _configure_map_data(self):
        # Load the park data
        fiona.supported_drivers['KML'] = 'rw'
        park_data = gpd.read_file(self.park_filepath, driver='KML')
        park_coordinates = list(park_data['geometry'][0].exterior.coords)
        self.park_polygon = Polygon(park_coordinates)
        if self.n_devices is not None:
                self.device_points = uc2_utils.generate_random_points_in_polygon(self.park_polygon, self.n_devices)
        else:
            self.device_points = uc2_utils.generate_evently_spaced_points_in_polygon(self.park_polygon, self.device_spacing)
        devices_locs = [(point.x, point.y) for point in self.device_points]
        self.park_lons, self.park_lats, *_ = zip(*park_coordinates)
        self.dev_lons, self.dev_lats = zip(*devices_locs)
        self.devices_frame = gpd.GeoDataFrame(devices_locs,  geometry=gpd.points_from_xy(self.dev_lons,self.dev_lats,crs='epsg:4326'))
        self.devices_frame['coordinatesUTM'] = self.devices_frame['geometry'].to_crs(epsg=32633)
        self.devices_frame['coverage'] = (self.devices_frame['coordinatesUTM'].buffer(self.device_trigger_radius)).to_crs(epsg=4326)
        self.devices_frame['status'] = 'inactive'
        self.devices_frame['activated'] = False
        
        # Hardcode fire point for now
        self.fire_point = uc2_utils.generate_random_points_in_polygon(self.park_polygon, 1)[0]
        self.fire_area_maj_ax = self.initial_fire_size
        self.fire_area_min_ax = self.initial_fire_size
        self.fire_area = uc2_utils.elliptical_buffer(self.fire_point, self.fire_area_maj_ax, self.fire_area_min_ax, self.wind_direction)
        self.smoke_area_maj_ax = self.initial_fire_size + self.smoke_area_offset
        self.smoke_area_min_ax = self.initial_fire_size + self.smoke_area_offset
        self.smoke_area = uc2_utils.elliptical_buffer(self.fire_point, self.smoke_area_maj_ax, self.smoke_area_min_ax, self.wind_direction)
    
    def _plot_fire_and_devices(self):
        self.bokeh_server.smoke_fire_area_present = True
        self.bokeh_server.devices_frame = self.devices_frame
        self.bokeh_server.fire_area_loc = self.fire_area
        self.bokeh_server.smoke_area_loc = self.smoke_area
        self.bokeh_server.update_requested = True
    
    def _create_plot(self):

        available_files = sorted([os.path.splitext(f)[0] for f in os.listdir(self.park_dir) if f.endswith('.kml')])
        
        map_options = GMapOptions(lat=self.park_polygon.centroid.y, lng=self.park_polygon.centroid.x, map_type="terrain", zoom=10)
        self.bokeh_server = BokehServerMap(self.api_key, 'Device activation status', map_options, available_files)
        self.bokeh_server.add_data(
            self.devices_frame,
            self.fire_point,
            self.park_polygon,
            self.fire_area,
            self.smoke_area,
        )
        self.bokeh_server.start_bokeh_server()


    def save_map(self):
        file_path = self.maps_save_path / f'map{self._map_idx}.html'
        self.bokeh_server.save_plot_state(file_path)
        self._map_idx += 1
        
    async def send_requests(self):
        offloaded_functions = []
        for i, device in self.devices_frame.iterrows():
            if device['status'] == 'fire':
                offloaded_functions.append(uc2_utils.offload_function(self.fire_img_file.absolute(), i))
            elif device['status'] == 'wood':
                offloaded_functions.append(uc2_utils.offload_function(self.wood_img_file.absolute(), i))
            
        results = await asyncio.gather(*offloaded_functions)
        for result in results:
            v, i = result
            if isinstance(v, str):
                logger.warning(f'Error encountered when offloading function for dev: {i}:\n{v}')
            else:
                self.devices_frame['activated'][i] = v
        
        
    async def spread_fire(self):
        for i in range(self.simulations_steps):
            logger.info(f'Simulation step {i+1}/{self.simulations_steps}')
            # Adjust fire center
            self.fire_point = uc2_utils.move_point(self.fire_point, self.fire_speed, self.wind_direction)
            
            # Create new fire area
            self.fire_area_maj_ax += self.fire_speed*2
            self.fire_area_min_ax += self.fire_speed
            self.fire_area = uc2_utils.elliptical_buffer(self.fire_point, self.fire_area_maj_ax, self.fire_area_min_ax, self.wind_direction)
            self.fire_area = self.fire_area.intersection(self.park_polygon)

            
            # Create new smoke area
            self.smoke_area_maj_ax += self.fire_speed*2
            self.smoke_area_min_ax += self.fire_speed
            self.smoke_area = uc2_utils.elliptical_buffer(self.fire_point, self.smoke_area_maj_ax, self.smoke_area_min_ax, self.wind_direction)
            self.smoke_area = self.smoke_area.intersection(self.park_polygon)
            
            # Check intersection with devices
            self.devices_frame['status'][self.devices_frame['coverage'].intersects(self.smoke_area)] = 'wood'
            self.devices_frame['status'][self.devices_frame['coverage'].intersects(self.fire_area)] = 'fire'

            # Activate said devices
            await self.send_requests()
            # Plot, if necessary
            if self.plotting:
                self._plot_fire_and_devices()
        self.bokeh_server.running = False

    def run(self):
        while True:
            self._update_settings_from_dashboard()
            if self.bokeh_server.running:
                asyncio.run(self.spread_fire())
            else:
                sleep(0.1)
            

if __name__ == '__main__':
    load_dotenv()
    ma = MapAsync(settings_file='settings.toml', plotting=True)
    ma.run()
