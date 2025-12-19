import logging
import threading

import numpy as np
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.document import Document
from bokeh.layouts import Spacer, column, row
from bokeh.models import (Button, ColumnDataSource, GMapOptions, Select,
                          Slider, TablerIcon)
from bokeh.plotting import figure, gmap, save
from bokeh.server.server import Server
from shapely.geometry import Polygon

logger = logging.getLogger('uc2')

class BokehServer:

    def __init__(self) -> None:
        self.update_requested = False
        self.port = 5006
        self.running = False
        self._initialized = False

    def _update_data_frames(self):
        raise NotImplementedError("This method should be implemented in subclasses")

    def _create_bokeh_app(self, doc: Document):
        raise NotImplementedError("This method should be implemented in subclasses")

    def is_initialized(self) -> bool:
        return self._initialized

    def update_plot(self):
        if self.update_requested:
            self._update_data_frames()
            self.update_requested = False

    def start_bokeh_server(self):

        # Create and start the Bokeh server
        self.bokeh_handler = FunctionHandler(self._create_bokeh_app)
        self.bokeh_app = Application(self.bokeh_handler)

        # Run Bokeh server in a separate thread
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Print info for the user
        logger.info("Bokeh visualization server started at http://localhost:%d", self.port)
        logger.info("Open this URL in a web browser to see the visualization")

    def run_server(self):
        self.server = Server({'/': self.bokeh_app}, port=self.port)
        self.server.start()
        self.server.io_loop.add_callback(self.server.show, "/")
        self.server.io_loop.start()

    def save_plot_state(self, filename: str):
        """Saves the current plot state to a file."""
        if not hasattr(self, 'doc'):
            raise RuntimeError("Bokeh document is not initialized. Call start_bokeh_server() first.")
        save(self.doc, filename)


class BokehServerStat(BokehServer):

    def __init__(self, matrix: np.ndarray) -> None:
        super().__init__()
        self.matrix = matrix
        self._matrix_side = matrix.shape[0]

    def _update_data_frames(self):
        self.source.data.update(image=[self.matrix])

    def _create_bokeh_app(self, doc):
        # Create the initial plot
        p = figure(title="Device Activation Status",
                   x_range=(0, self._matrix_side),
                   y_range=(0, self._matrix_side),
                   tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])

        # Create a color mapper
        self.source = ColumnDataSource(data=dict(
            image=[self.matrix],
            x=[0], y=[0], dw=[self._matrix_side], dh=[self._matrix_side]
        ))

        # Add image renderer
        p.image(source=self.source, image='image', x='x', y='y', dw='dw', dh='dh', palette=["green", "red"])

        doc.add_root(p)

        # Set up periodic callback to check for updates

        doc.add_periodic_callback(self.update_plot, 100)
        self.running = True

        self.doc = doc
        self._initialized = True


class BokehServerMap(BokehServer):

    def __init__(self, api_key: str, title: str, map_options: GMapOptions, map_names: list[str]) -> None:
        super().__init__()

        self.api_key = api_key
        self.title = title
        self.map_options = map_options
        self.map_names = map_names
        self.devices_frame = None
        self.fire_point = None
        self.park_loc = None
        self.fire_area_loc = None
        self.smoke_area_loc = None
        self.timings = None
        self.smoke_fire_area_present = False
        self.data_frames_setup = False
        self.update_park = False
        
    def add_data(self, devices_frame, fire_point, park_loc, fire_area_loc, smoke_area_loc, timings):
        """Method to add data to the server."""
        self.devices_frame = devices_frame
        self.fire_point = fire_point
        self.park_loc = park_loc
        self.fire_area_loc = fire_area_loc
        self.smoke_area_loc = smoke_area_loc
        self.timings = timings
        if not self.data_frames_setup:
            self.setup_data_frames()

    def setup_data_frames(self):
        self.devices = ColumnDataSource(data=dict(
            lon=self.devices_frame[0],
            lat=self.devices_frame[1],
            color=['green'] * len(self.devices_frame)
        ))
        self.device_area = ColumnDataSource(data=dict(
            lon = [cov.exterior.coords.xy[0].tolist() for cov in self.devices_frame['coverage']],
            lat = [cov.exterior.coords.xy[1].tolist() for cov in self.devices_frame['coverage']],
        ))
        self.activated_devices = ColumnDataSource(data=dict(
            x = [0],
            y = [0],
        ))
        self.timings_source = ColumnDataSource(data=dict(
            x = list(range(len(self.timings))),
            y = self.timings,
        ))
        self.fire = ColumnDataSource(data=dict(
            lon=[self.fire_point.x],
            lat=[self.fire_point.y],
        ))
        self.park = ColumnDataSource(data=dict(
            lon=self.park_loc.exterior.coords.xy[0].tolist(),
            lat=self.park_loc.exterior.coords.xy[1].tolist(),
        ))
        self.fire_area = ColumnDataSource(data=dict(
            lon=self.fire_area_loc.exterior.coords.xy[0].tolist(),
            lat=self.fire_area_loc.exterior.coords.xy[1].tolist(),
        ))
        self.smoke_area = ColumnDataSource(data=dict(
            lon=self.smoke_area_loc.exterior.coords.xy[0].tolist(),
            lat=self.smoke_area_loc.exterior.coords.xy[1].tolist(),
        ))
        self.data_frames_setup = True
        
    def update_map_data(self, map_options = None, park = None, fire_point = None, device_data = None):
        if map_options is not None:
            self.map_options = map_options
        if park is not None:
            self.park_loc = park
        if fire_point is not None:
            self.fire_point = fire_point
        if device_data is not None:
            self.devices_frame = device_data
        
        self.update_park = True
        self.update_requested = True

    def _update_data_frames(self):

        colors = ['red' if device['activated'] else 'green' for _, device in self.devices_frame.iterrows()]

        self.devices.data.update(
            lon=self.devices_frame[0],
            lat=self.devices_frame[1],
            color=colors
        )
        history_active_devices = self.activated_devices.data['y']
        history_active_devices.append(int(self.devices_frame['activated'].sum()))
        x_activated_devices = list(range(len(history_active_devices)))
        self.activated_devices.data.update(
            x=x_activated_devices,
            y=history_active_devices,
        )
        self.timings_source.data.update(
            x = list(range(len(self.timings))),
            y = self.timings,
        )
        alpha = 0.2 if self.smoke_fire_area_present else 0.0
        if isinstance(self.fire_area_loc, Polygon):
            self.fire_area.data.update(
                lon=self.fire_area_loc.exterior.coords.xy[0].tolist(),
                lat=self.fire_area_loc.exterior.coords.xy[1].tolist(),
            )
        if isinstance(self.smoke_area_loc, Polygon):
            self.smoke_area.data.update(
                lon=self.smoke_area_loc.exterior.coords.xy[0].tolist(),
                lat=self.smoke_area_loc.exterior.coords.xy[1].tolist(),
            )
        self.fire_area_patch.glyph.fill_alpha = alpha*1.5
        self.smoke_area_patch.glyph.fill_alpha = alpha
        
        if self.update_park:
            self.map_plot.map_options.lat = self.map_options.lat
            self.map_plot.map_options.lng = self.map_options.lng
            self.device_area.data.update(
                lon = [cov.exterior.coords.xy[0].tolist() for cov in self.devices_frame['coverage']],
                lat = [cov.exterior.coords.xy[1].tolist() for cov in self.devices_frame['coverage']],
            )                          
            self.fire.data.update(
                lon=[self.fire_point.x],
                lat=[self.fire_point.y],
            )
            self.park.data.update(
                lon=self.park_loc.exterior.coords.xy[0].tolist(),
                lat=self.park_loc.exterior.coords.xy[1].tolist(),
            )
            self.activated_devices.data.update(
                x = [0],
                y = [0],
            )
            self.update_park = False

    def _start_button_click(self):
        self.running = True

    def _create_bokeh_app(self, doc):
        # title_div = Div(text=f'<h1>{self.title}</h1>', styles={'text-align': 'center'})
        self.map_plot = gmap(self.api_key, self.map_options,
                        title='Devices map with fire area',
                        sizing_mode='scale_both',
                        height_policy='max',
                        # width_policy='max',
                        min_width=700,
                        )
        self.map_plot.patches(source=self.device_area, xs='lon', ys='lat', fill_color='teal', alpha=0.2)
        self.map_plot.scatter(source=self.devices, x='lon', y='lat', fill_color='color', size=10)
        self.map_plot.scatter(source=self.fire, x='lon', y='lat', fill_color='red', size=10)
        self.map_plot.patch(source=self.park, x='lon', y='lat', fill_color='cornflowerblue', alpha=0.2)
        self.fire_area_patch = self.map_plot.patch(source=self.fire_area, x='lon', y='lat', fill_color='red', alpha=0.0)
        self.smoke_area_patch = self.map_plot.patch(source=self.smoke_area, x='lon', y='lat', fill_color='yellow', alpha=0.0)
        line_plot = figure(title='Active Devices Over Time',
                           x_axis_label='Step',
                           y_axis_label='Active Devices',
                           sizing_mode='scale_width',
                           height=400,
                           )
        timing_plot = figure(title='Offloading Time',
                           x_axis_label='Step',
                           y_axis_label='Offloading Time',
                           sizing_mode='scale_width',
                           height=400,
                           )
        blank_spacer = Spacer(height_policy='max')
        self.map_name_select = Select(title="Map name", value=self.map_names[0], options=self.map_names)
        self.wind_direction_slider = Slider(title="Wind direction [°N]", value=40, start=0, end=350, step=10)
        self.wind_speed_slider = Slider(title="Wind speed [m/s]", value=5, start=1, end=20, step=1)
        self.start_button = Button(icon=TablerIcon('player-play'), label='Start', button_type='success', min_width=300, max_width=500, width_policy='max', margin=(50,10,50,0))
        self.start_button.on_click(self._start_button_click)
        c = column([line_plot, timing_plot, self.map_name_select, self.wind_direction_slider, self.wind_speed_slider, blank_spacer, self.start_button],
                   sizing_mode='scale_width',
                   width_policy='min',
                   min_width=300,
                   max_width=500,
                   )
        self.line = line_plot.vbar(x='x', top='y', source=self.activated_devices, width=0.5, color='green')
        self.timing_line = timing_plot.vbar(x='x', top='y', source=self.timings_source, width=0.5, color='blue')
        grid_plot = row([self.map_plot,c],sizing_mode='scale_both')
        # page_layout = column(title_div, grid_plot, sizing_mode='scale_both')
        doc.add_root(grid_plot)

        # Set up periodic callback to check for updates
        doc.add_periodic_callback(self.update_plot, 100)

        self.doc = doc
        self._initialized = True
