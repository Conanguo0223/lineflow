import pygame
import numpy as np
import simpy
import warnings

from lineflow.helpers import (
    zip_cycle,
    compute_performance_coefficient,
)

from lineflow.simulation.states import (
    TokenState,
    DiscreteState,
    CountState,
    ObjectStates,
    NumericState,
)
from lineflow.simulation.stationary_objects import StationaryObject
from lineflow.simulation.connectors import Buffer
from lineflow.simulation.movable_objects import (
    Part,
    Carrier,
    Worker,
)

from collections import deque
import bisect


class WorkerPool(StationaryObject):

    def __init__(
        self,
        name,
        n_workers=None,
        transition_time=5,
        use_rates=False,
        use_normalization=False
    ):
        super().__init__()

        assert n_workers is not None, "Workers have to be set"

        self.use_rates = use_rates
        self.use_normalization = use_normalization
        self.name = name
        self.n_workers = n_workers
        self.transition_time = transition_time
        self.stations = []
        self._station_names = []
        self._worker_names = [f"W{i}" for i in range(self.n_workers)]
        self.workers = {
            name: Worker(
                name=name,
                transition_time=self.transition_time
            ) for name in self._worker_names
        }

    def register_station(self, station):
        self.stations.append(station)
        self._station_names.append(station.name)

    def get_connected_stations_throughput_rates(self):
        """Get throughput rates of all connected stations"""
        throughput_rates = []
        
        for station in self.stations:
            if hasattr(station, 'state') and 'throughput_rate' in station.state.names:
                throughput_rate = station.state['throughput_rate'].value
                throughput_rates.append(throughput_rate)
            else:
                # If station doesn't have throughput_rate, default to 0
                throughput_rates.append(0.0)
        
        return throughput_rates

    def get_average_throughput_rate(self):
        """Get average throughput rate of all connected stations"""
        throughput_rates = self.get_connected_stations_throughput_rates()
        
        if throughput_rates:
            return sum(throughput_rates) / len(throughput_rates)
        else:
            return 0.0
        
    def _normalize_n_workers(self, n_workers, max_workers=15.0):
        """Normalize number of workers to [0, 1] range"""
        return min(n_workers / max_workers, 1.0)

    def _normalize_transition_time(self, transition_time, max_transition_time=40.0):
        """Normalize transition time to [0, 1] range"""
        return min(transition_time / max_transition_time, 1.0)

    def _normalize_n_stations(self, n_stations, max_stations=5.0):
        """Normalize number of stations to [0, 1] range"""
        return min(n_stations / max_stations, 1.0)

    def _update_normalized_states(self):
        """Update normalized states based on current discrete/numeric states"""
        if not self.use_normalization:
            return
            
        # Update normalized n_workers if it exists
        if 'norm_n_workers' in self.state.names and 'n_workers' in self.state.names:
            current_n_workers = self.state['n_workers'].value
            normalized_n_workers = self._normalize_n_workers(current_n_workers)
            self.state['norm_n_workers'].update(normalized_n_workers)
        
        # Update normalized transition_time if it exists
        if 'norm_transition_time' in self.state.names and 'transition_time' in self.state.names:
            current_transition_time = self.state['transition_time'].value
            normalized_transition_time = self._normalize_transition_time(current_transition_time)
            self.state['norm_transition_time'].update(normalized_transition_time)

        # Update normalized n_stations if it exists
        if 'norm_n_stations' in self.state.names and 'n_stations' in self.state.names:
            current_n_stations = self.state['n_stations'].value
            normalized_n_stations = self._normalize_n_stations(current_n_stations)
            self.state['norm_n_stations'].update(normalized_n_stations)

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        4 observables
        - n_workers                     (observable)
        - transition_time               (observable)
        - n_stations                    (observable)
        - assigned_station*n_workers    (observable)

        rate-wise states:
        6 observables
        - n_workers                     (observable)
        - transition_time               (observable)
        - n_stations                    (observable)
        - assigned_station*n_workers    (non-observable)
        --------------------------------------------------------------
        - avg_throughput_rate           (observable)

        """
        for worker in self.workers.values():
            if self.use_rates:
                worker.init_state(self.stations, is_observable=False)
            else:
                worker.init_state(self.stations)

        if self.use_rates and self.use_normalization == False:
            self.state = ObjectStates(
                *[
                    worker.state for worker in self.workers.values()
                ],CountState('n_workers', is_actionable=False, is_observable=True, vmin=0),
                NumericState('transition_time', is_actionable=False, is_observable=True, vmin=0),
                CountState('n_stations', is_actionable=False, is_observable=True, vmin=0),
                NumericState('avg_throughput_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        elif self.use_normalization and self.use_rates:
            self.state = ObjectStates(
                *[
                    worker.state for worker in self.workers.values()
                ],CountState('n_workers', is_actionable=False, is_observable=False, vmin=0),
                NumericState('norm_n_workers', is_actionable=False, is_observable=True, vmin=0.0, vmax=1.0),
                NumericState('transition_time', is_actionable=False, is_observable=False, vmin=0),
                NumericState('norm_transition_time', is_actionable=False, is_observable=True, vmin=0.0, vmax=1.0),
                CountState('n_stations', is_actionable=False, is_observable=False, vmin=0),
                NumericState('norm_n_stations', is_actionable=False, is_observable=True, vmin=0.0, vmax=1.0),
                NumericState('avg_throughput_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        else:
            self.state = ObjectStates(
                *[
                    worker.state for worker in self.workers.values()
                ],CountState('n_workers', is_actionable=False, is_observable=True, vmin=0),
                NumericState('transition_time', is_actionable=False, is_observable=True, vmin=0),
                CountState('n_stations', is_actionable=False, is_observable=True, vmin=0),
            )
        self.state['transition_time'].update(self.transition_time)
        self.state['n_workers'].update(self.n_workers)
        self.state['n_stations'].update(self.n_stations)
        if self.use_rates:
            self.state['avg_throughput_rate'].update(0.0)
            if self.use_normalization:
                self.state['norm_n_workers'].update(self._normalize_n_workers(self.n_workers))
                self.state['norm_transition_time'].update(self._normalize_transition_time(self.transition_time))
                self.state['norm_n_stations'].update(self._normalize_n_stations(self.n_stations))
        # Distribute worker on stations in round robin fashion
        for worker, station in zip_cycle(self.n_workers, self.n_stations):
            self.state[f"W{worker}"].apply(station)

    def update_avg_throughput_rate(self):
        """Update the average throughput rate of connected stations"""
        if 'avg_throughput_rate' in self.state.names:
            avg_rate = self.get_average_throughput_rate()
            self.state['avg_throughput_rate'].update(avg_rate)

    @property
    def n_stations(self):
        return len(self.stations)

    def register(self, env):
        self.env = env

        for worker in self.workers.values():
            worker.register(env)

        for worker_n, station in zip_cycle(self.n_workers, self.n_stations):

            worker = self.workers[f"W{worker_n}"]
            # Start working
            self.env.process(worker.work())
        # if self.use_rates:
        #     self.env.process(self._throughput_monitoring_loop())

    def _throughput_monitoring_loop(self):
        """Continuously update average throughput metrics"""
        while True:
            if hasattr(self, 'state'):
                self.update_avg_throughput_rate()
            yield self.env.timeout(1)  # Update every simulation time unit
            
    def apply(self, actions):
        """
        This should just update the state of the workers
        """
        for worker, station in actions.items():
            worker_obj = self.workers[worker]
            self.env.process(worker_obj.assign(station))
            # Update metrics only when events occur
            if self.use_rates:
                self.update_avg_throughput_rate()

    def get_worker(self, name):
        # gather these workers assigned to these station
        station = self._station_names.index(name)
        requests = {}

        for worker in self.workers.values():
            # If state of worker equals the station, the worker is blocked for exactly this station
            if worker.state.value == station:
                requests[worker.name] = worker
        return requests
    
    def get_connected_stations_throughput_rates(self):
        """Get throughput rates of all connected stations"""
        throughput_rates = []
        
        for station in self.stations:
            if hasattr(station, 'state') and 'throughput_rate' in station.state.names:
                throughput_rate = station.state['throughput_rate'].value
                throughput_rates.append(throughput_rate)
            else:
                # If station doesn't have throughput_rate, default to 0
                throughput_rates.append(0.0)
        
        return throughput_rates

    def get_average_throughput_rate(self):
        """Get average throughput rate of all connected stations"""
        throughput_rates = self.get_connected_stations_throughput_rates()
        
        if throughput_rates:
            return sum(throughput_rates) / len(throughput_rates)
        else:
            return 0.0


class Station(StationaryObject):

    _width = 30
    _height = 30
    _color = 'black'

    def __init__(
        self,
        name,
        position=None,
        processing_time=5,
        processing_std=None,
        rework_probability=0,
        worker_pool=None,
        use_rates=False,  # test out whether to use rates
        use_normalization=False
    ):

        super().__init__()

        if position is None:
            position = (0, 0)

        self.name = name
        self.position = pygame.Vector2(position[0], position[1])

        self.worker_pool = worker_pool
        self.worker_requests = {}
        self.use_rates = use_rates
        self.use_normalization = use_normalization
        self.state_ = []
        if self.worker_pool is not None:
            self.worker_pool.register_station(self)

        self.processing_time = processing_time
        self.rework_probability = rework_probability

        if self.rework_probability > 1 or self.rework_probability < 0:
            raise ValueError('rework_probability should should be between 0 and 1')

        if processing_std is None:

            self.processing_std = 0.1*self.processing_time
        else:
            assert processing_std >= 0 and processing_std <= 1
            self.processing_std = processing_std*self.processing_time

        self.worker_assignments = {}
        # Time-based throughput tracking
        self.time_based_window_size = 50  # Time window size for moving average
        max_events = int(self.time_based_window_size * 20)  # Keep 20 windows worth
        self.throughput_events = deque(maxlen=max_events)
        self.moving_window_lookback = 5  # Number of window periods to average
        self.utilization_events = deque(maxlen=max_events)  # List of utilization events

        # Cache for expensive calculations
        self._throughput_cache = None
        self._cache_timestamp = 0
        self._cache_validity = 5.0  # Cache valid for 5 time units

    @property
    def is_automatic(self):
        return self.worker_pool is None

    @property
    def n_workers(self):
        if self.worker_pool is not None:
            return len(self.worker_assignments) + 1
        else:
            return 1
    @property
    def n_workers_proportional(self):
        if self.worker_pool is not None:
            return len(self.worker_assignments) / self.worker_pool.n_workers
        else:
            return 1

    def setup_draw(self):

        self._rect = pygame.Rect(
            self.position.x - self._width / 2,
            self.position.y - self._height / 2,
            self._width,
            self._height,
        )

        font = pygame.font.SysFont(None, 20)
        self._text = font.render(self.name, True, 'black')

    def _draw(self, screen):
        pygame.draw.rect(screen, self._color, self._rect, border_radius=5)
        self._draw_info(screen)
        screen.blit(
            self._text,
            self._text.get_rect(center=self.position + (0, -0.6 * self._height)),
        )

    def _draw_info(self, screen):
        pass

    def _draw_n_workers(self, screen):
        if not self.is_automatic:
            font = pygame.font.SysFont(None, 14)
            text = font.render(
                "W=" + str(self.n_workers),
                True,
                'black',
            )
            screen.blit(
                text,
                text.get_rect(center=self.position),
            )

    def _draw_n_carriers(self, screen):
        font = pygame.font.SysFont(None, 14)
        text = font.render(
            "C=" + self.state['carriers_in_magazine'].to_str(),
            True,
            'black',
        )
        screen.blit(
            text,
            text.get_rect(center=self.position),
        )

    def get_performance_coefficient(self):
        return compute_performance_coefficient(self.n_workers)

    def _sample_exp_time(self, time=None, scale=None, rework_probability=0):
        """
        Samples a time from an exponential distribution
        """

        coeff = self.get_performance_coefficient()

        t = time*coeff + self.random.exponential(scale=scale)

        rework = self.random.choice(
            [1, 2],
            p=[1-rework_probability, rework_probability],
        )

        return t*rework

    def set_to_waiting(self):
        yield self.env.timeout(0)
        self._color = 'yellow'
        self.state['mode'].update('waiting')
        if self.use_normalization:
            self._update_normalized_states()
        yield self.env.timeout(0)

    def request_workers(self):
        """
        Requests (and blocks) the worker for the process coming up.
        """
        if not self.is_automatic:
            self.worker_assignments = self.worker_pool.get_worker(self.name)

            self.worker_requests = {
                name: worker.request() for name, worker in self.worker_assignments.items()
            }

            # Block workers for this process
            for request in self.worker_requests.values():
                yield request

        else:
            yield self.env.timeout(0)

    def release_workers(self):
        """
        Releases the worker, to they may follow a new assignment
        """
        if not self.is_automatic:

            for worker, request in self.worker_requests.items():
                self.worker_assignments[worker].release(request)
            self.worker_requests = {}
            self.worker_assignments = {}

    def set_to_error(self):
        yield self.env.timeout(0)
        self._color = 'red'
        self.state['mode'].update('failing')
        if self.use_normalization:
            self._update_normalized_states()
        yield self.env.timeout(0)

    def set_to_work(self):
        yield self.env.timeout(0)
        self._color = 'green'
        self.state['mode'].update('working')
        if self.use_normalization:
            self._update_normalized_states()
        yield self.env.timeout(0)

    def turn_off(self):
        self._color = 'gray'
        self.state['on'].update(False)
        self.turn_off_event = simpy.Event(self.env)
        return self.turn_off_event

    def is_on(self):
        return self.state['on'].to_str()

    def turn_on(self):
        event = self.turn_off_event

        self.state['on'].update(True)
        if not event.triggered:
            yield event.succeed()
        else:
            yield self.env.timeout(0)

    def connect_to_input(self, station, *args, **kwargs):
        buffer = Buffer(name=f"Buffer_{station.name}_to_{self.name}", *args, **kwargs)
        self._connect_to_input(buffer)
        station._connect_to_output(buffer)

    def connect_to_output(self, station, *args, **kwargs):
        buffer = Buffer(name=f"Buffer_{self.name}_to_{station.name}", *args, **kwargs)
        self._connect_to_output(buffer)
        station._connect_to_input(buffer)

    def _connect_to_input(self, buffer):
        if hasattr(self, 'buffer_in'):
            raise ValueError(f'Input of {self.name} already connected')
        self.buffer_in = buffer.connect_to_output(self)

    def _connect_to_output(self, buffer):
        if hasattr(self, 'buffer_out'):
            raise ValueError(f'Output of {self.name} already connected')
        self.buffer_out = buffer.connect_to_input(self)

    def register(self, env):
        self.env = env
        self.env.process(self.run())

        # Start continuous throughput monitoring
        # self.env.process(self._throughput_monitoring_loop())
    
    def _throughput_monitoring_loop(self):
        """Continuously update throughput metrics"""
        while True:
            if hasattr(self, 'state'):
                self._update_throughput_metrics()
            yield self.env.timeout(1)  # Update every simulation time unit

    def _derive_actions_from_new_state(self, state):
        # Turn machine back on if needed
        if not self.is_on() and 'on' in state and hasattr(self, 'turn_off_event') and state['on'] == 0:
            self.env.process(self.turn_on())

    def apply(self, actions):
        self._derive_actions_from_new_state(actions)
        self.state.apply(actions)

    def _record_utilization_time_based(self, mode):
        current_time = self.env.now
        self.utilization_events.append({
            'timestamp': current_time,
            'mode': mode,
        })
        # Keep only events within reasonable  (e.g., last 10 windows)
        cutoff_time = current_time - (self.time_based_window_size * 10)
        self.utilization_events = [
            event for event in self.utilization_events
            if event['timestamp'] > cutoff_time
        ]

    def _record_throughput_time_based(self, carrier):
        """Record object passing through the station"""
        current_time = self.env.now
        
        # Count parts on carrier
        part_count = 1
        if hasattr(carrier, 'parts'):
            part_count = len(carrier.parts)
        
        # Add event to 
        self.throughput_events.append({
            'timestamp': current_time,
            'carriers': 1,
            'parts': part_count
        })
        
        # Keep only events within reasonable  (e.g., last 10 windows)
        cutoff_time = current_time - (self.time_based_window_size * 10)
        self.throughput_events = [
            event for event in self.throughput_events 
            if event['timestamp'] > cutoff_time
        ]
        # Update metrics only when events occur
        if self.use_rates:
            self._update_throughput_metrics()
    def _get_moving_window_utilization(self, window_end_time=None):
        """Calculate utilization rate for a moving window"""
        if window_end_time is None:
            window_end_time = self.env.now
        
        window_start_time = window_end_time - self.time_based_window_size
        
        # Find events within the window
        window_events = [
            event for event in self.utilization_events
            if window_start_time <= event['timestamp'] <= window_end_time
        ]
        
        if not window_events:
            return {'utilization_rate': 0.0}
        
        # Calculate time spent in each mode
        mode_durations = {'working': 0.0, 'waiting': 0.0, 'failing': 0.0}
        
        # Add starting point if needed
        if window_events[0]['timestamp'] > window_start_time:
            # Assume 'waiting' mode before first event
            prev_mode = 'waiting'
            prev_time = window_start_time
        else:
            prev_mode = window_events[0]['mode']
            prev_time = window_events[0]['timestamp']
        
        # Calculate durations
        for event in window_events:
            duration = event['timestamp'] - prev_time
            if prev_mode in mode_durations:
                mode_durations[prev_mode] += duration
            prev_time = event['timestamp']
            prev_mode = event['mode']
        
        # Add final duration to window end
        final_duration = window_end_time - prev_time
        if prev_mode in mode_durations:
            mode_durations[prev_mode] += final_duration
        
        # Calculate utilization rate
        total_time = sum(mode_durations.values())
        utilization_rate = mode_durations['working'] / total_time if total_time > 0 else 0.0
        
        return {
            'utilization_rate': utilization_rate,
            'working_time': mode_durations['working'],
            'total_time': total_time,
            'window_start': window_start_time,
            'window_end': window_end_time
        }

    def _get_moving_window_throughput(self, window_end_time=None):
        """Calculate throughput for a moving window ending at specified time"""
        if window_end_time is None:
            window_end_time = self.env.now
        
        # Check cache validity
        cache_key = window_end_time
        if (self._throughput_cache is not None and 
            self._cache_timestamp == cache_key and
            (self.env.now - self._cache_timestamp) < self._cache_validity):
            return self._throughput_cache

        window_start_time = window_end_time - self.time_based_window_size
        # Use deque - much faster than list comprehension
        total_carriers = 0
        total_parts = 0
        for event in self.throughput_events:
            if window_start_time <= event['timestamp'] <= window_end_time:
                total_carriers += event['carriers']
                total_parts += event['parts']
        
        # Calculate rates (per time unit)
        carrier_rate = total_carriers / self.time_based_window_size
        parts_rate = total_parts / self.time_based_window_size
        
        result = {
            'carriers_in_window': total_carriers,
            'parts_in_window': total_parts,
            'carrier_rate': carrier_rate,
            'parts_rate': parts_rate,
            'window_start': window_start_time,
            'window_end': window_end_time
        }
        
        # Cache result
        self._throughput_cache = result
        self._cache_timestamp = cache_key
        
        return result

    def _get_moving_average_throughput(self, num_windows=None):
        """Calculate moving average over multiple window periods"""
        if num_windows is None:
            num_windows = self.moving_window_lookback
        
        current_time = self.env.now
        window_rates = []
        
        # Calculate throughput for each of the last N windows
        for i in range(num_windows):
            window_end = current_time - (i * self.time_based_window_size)
            if window_end >= self.time_based_window_size:  # Ensure we have a full window
                window_data = self._get_moving_window_throughput(window_end)
                window_rates.append(window_data['carrier_rate'])
        
        # Return average rate
        if window_rates:
            return sum(window_rates) / len(window_rates)
        else:
            return 0.0

    def _update_throughput_metrics(self):
        """Update moving window throughput metrics"""
        if not hasattr(self, 'state'):
            return
        
        # Get current moving window data
        current_window = self._get_moving_window_throughput()
        
        # Update current window throughput count
        if 'current_window_throughput' in self.state.names:
            self.state['current_window_throughput'].update(current_window['carriers_in_window'])
        
        # Update current throughput rate
        if 'throughput_rate' in self.state.names:
            self.state['throughput_rate'].update(current_window['carrier_rate'])
        
        # Update moving average throughput
        if 'avg_throughput_last_5_windows' in self.state.names:
            avg_rate = self._get_moving_average_throughput(5)
            self.state['avg_throughput_last_5_windows'].update(avg_rate)

        if 'current_work_in_process' in self.state.names:
            self.state['current_work_in_process'].update(current_window['parts_in_window'])

        # get utilization 
        utilization_data = self._get_moving_window_utilization()
        if 'utilization_rate' in self.state.names:
            self.state['utilization_rate'].update(utilization_data['utilization_rate'])
        if 'working_time' in self.state.names:
            self.state['working_time'].update(utilization_data['working_time'])
        if 'total_time' in self.state.names:
            self.state['total_time'].update(utilization_data['total_time'])


    def get_throughput_for_period(self, start_time, end_time):
        """Get throughput data for a specific time period"""
        period_events = [
            event for event in self.throughput_events
            if start_time <= event['timestamp'] <= end_time
        ]
        
        total_carriers = sum(event['carriers'] for event in period_events)
        total_parts = sum(event['parts'] for event in period_events)
        duration = end_time - start_time
        
        return {
            'total_carriers': total_carriers,
            'total_parts': total_parts,
            'carrier_rate': total_carriers / duration if duration > 0 else 0,
            'parts_rate': total_parts / duration if duration > 0 else 0,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }

    def get_recent_throughput_windows(self, num_windows=5):
        """Get throughput data for the last N moving windows"""
        current_time = self.env.now
        windows = []
        
        for i in range(num_windows):
            window_end = current_time - (i * self.time_based_window_size)
            if window_end >= self.time_based_window_size:
                window_data = self._get_moving_window_throughput(window_end)
                windows.append(window_data)
        
        return list(reversed(windows))  # Return in chronological order
    
    def get_recent_throughput(self, time_periods=5):
        """Get throughput for the last N time periods"""
        return self.throughput_[-time_periods:] if len(self.throughput_) >= time_periods else self.throughput_

    def _normalize_mode(self, mode):
        """Normalize mode to [0, 1] range"""
        mode_mapping = {
            'waiting': 0.0,
            'working': 0.5,
            'failing': 1.0
        }
        return mode_mapping.get(mode, 0.0)

    def _normalize_processing_time(self, processing_time, max_processing_time=100.0):
        """Normalize processing time to [0, 1] range"""
        return min(processing_time / max_processing_time, 1.0)

    def _normalize_waiting_time(self, waiting_time, max_waiting_time=100.0):
        """Normalize waiting time to [0, 1] range"""
        return min(waiting_time / max_waiting_time, 1.0)

    def _update_normalized_states(self):
        """Update normalized states based on current discrete/numeric states"""
        if not self.use_normalization:
            return
            
        # Update normalized mode if it exists
        if 'norm_mode' in self.state.names and 'mode' in self.state.names:
            current_mode = self.state['mode'].value
            normalized_mode = self._normalize_mode(current_mode)
            self.state['norm_mode'].update(normalized_mode)
        
        # Update normalized processing time if it exists
        if 'norm_processing_time' in self.state.names and 'processing_time' in self.state.names:
            current_processing_time = self.state['processing_time'].value
            normalized_processing_time = self._normalize_processing_time(current_processing_time)
            self.state['norm_processing_time'].update(normalized_processing_time)

        # Update normalized waiting time if it exists
        if 'norm_waiting_time' in self.state.names and 'waiting_time' in self.state.names:
            current_waiting_time = self.state['waiting_time'].value
            normalized_waiting_time = self._normalize_waiting_time(current_waiting_time)
            self.state['norm_waiting_time'].update(normalized_waiting_time)

class Assembly(Station):
    """
    Assembly takes a carrier from `buffer_in` and `buffer_component`, puts the parts of the component
    carrier on the carrier that came from buffer_in, and pushes that carrier to buffer_out and
    pushes the component carrier to buffer_return if a buffer return exists, otherwise these
    carriers are lost.

    Args:
        name (str): Name of the station
        processing_time (float): Time until parts are moved from component carrier to main carrier
        position (tuple): X and Y position in the visualization
        buffer_return (lineflow.simulation.connectors.Buffer): The buffer to
            put the old component carriers on
        processing_std (float): The standard deviation of the processing time
    """
    def __init__(
        self,
        name,
        buffer_in=None,
        buffer_out=None,
        buffer_component=None,
        processing_time=5,
        position=None,
        buffer_return=None,
        processing_std=None,
        NOK_part_error_time=2,
        worker_pool=None,
        use_rates=False,
        use_normalization=False,
    ):

        super().__init__(
            name=name,
            position=position,
            processing_time=processing_time,
            processing_std=processing_std,
            worker_pool=worker_pool,
            use_rates=use_rates,
            use_normalization=use_normalization,
        )
        max_events = int(self.time_based_window_size * 20)
        self.NOK_part_error_time = NOK_part_error_time
        self.scrap_events = deque(maxlen=max_events) 
        self.scrap_window_size = self.time_based_window_size
        self.scrap_moving_window_lookback = 5

        if buffer_in is not None:
            self._connect_to_input(buffer_in)

        if buffer_out is not None:
            self._connect_to_output(buffer_out)

        if buffer_component is not None:
            self.buffer_component = buffer_component.connect_to_output(self)

        if buffer_return is not None:
            self.buffer_return = buffer_return.connect_to_input(self)

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        4 observables
        - on                    (non-observable)
        - mode                  (observable)
        - carrier               (non-observable)
        - carrier_component     (non-observable)
        - n_scrap_parts         (observable)
        - n_workers             (observable)
        - processing_time       (observable)
        
        rate-wise states:
        6 observables
        - on                    (non-observable)
        - mode                  (observable)
        - carrier               (non-observable)
        - carrier_component     (non-observable)
        - n_scrap_parts         (non-observable)
        - n_workers             (non-observable)
        --------------------------------------------------------------
        - processing_time       (observable) 
        - scrap_rate            (observable)        org:n_scrap_parts
        - n_workers_proportion  (observable)        org:n_workers       
        - throughput_rate       (observable)        
        - utilization_rate      (observable)        
        """
        if self.use_rates and self.use_normalization == False:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),# OBS
                TokenState(name='carrier', is_observable=False),
                TokenState(name='carrier_component', is_observable=False),
                CountState('n_scrap_parts', is_actionable=False, is_observable=False),
                CountState('n_workers', is_actionable=False, is_observable=False, vmin=0),
                NumericState('processing_time', is_actionable=False, is_observable=True, vmin=0),# OBS
                # rate specific metrics
                NumericState('scrap_rate', is_actionable=False, is_observable=True, vmin=0),# OBS
                NumericState('n_workers_proportion', is_actionable=False, is_observable=True, vmin=0),# OBS
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),# OBS
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),# OBS
            )
        elif self.use_rates and self.use_normalization:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing'], is_observable=False),
                NumericState('norm_mode', is_actionable=False, is_observable=True, vmin=0),# Normed OBS
                TokenState(name='carrier', is_observable=False),
                TokenState(name='carrier_component', is_observable=False),
                CountState('n_scrap_parts', is_actionable=False, is_observable=False),
                CountState('n_workers', is_actionable=False, is_observable=False, vmin=0),
                NumericState('processing_time', is_actionable=False, is_observable=False, vmin=0),
                NumericState('norm_processing_time', is_actionable=False, is_observable=True, vmin=0),# Normed OBS
                # rate specific metrics
                NumericState('scrap_rate', is_actionable=False, is_observable=True, vmin=0),# OBS
                NumericState('n_workers_proportion', is_actionable=False, is_observable=True, vmin=0),# OBS
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),# OBS
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),# OBS
            )
        else:
            # uses original states
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='carrier_component', is_observable=False),
                CountState('n_scrap_parts', is_actionable=False, is_observable=True),
                CountState('n_workers', is_actionable=False, is_observable=True, vmin=0),
                NumericState('processing_time', is_actionable=False, is_observable=True, vmin=0),
            )
        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)
        self.state['carrier_component'].update(None)
        self.state['n_scrap_parts'].update(0)
        self.state['processing_time'].update(self.processing_time)
        self.state['n_workers'].update(self.n_workers)
        if self.use_rates:
            # rate specific metric
            self.state['scrap_rate'].update(0.0)
            self.state['n_workers_proportion'].update(self.n_workers_proportional)
            self.state['throughput_rate'].update(0.0)
            self.state['utilization_rate'].update(0.0)
            if self.use_normalization:
                self.state['norm_mode'].update(0.0)
                self.state['norm_processing_time'].update(0.0)
            

    def connect_to_component_input(self, station, *args, **kwargs):
        buffer = Buffer(name=f"Buffer_{station.name}_to_{self.name}", *args, **kwargs)
        self.buffer_component = buffer.connect_to_output(self)
        station._connect_to_output(buffer)

    def connect_to_component_return(self, station, *args, **kwargs):
        buffer = Buffer(name=f"Buffer_{self.name}_to_{station.name}", *args, **kwargs)
        self.buffer_return = buffer.connect_to_input(self)
        station._connect_to_input(buffer)

    def _has_invalid_components_on_carrier(self, carrier):
        """
        Checks if any of the components on the carrier is not valid for assembly. In this case,
        `True` is returned. Otherwise, `False` is returned.
        """
        for component in carrier:
            if not component.is_valid_for_assembly(self.name):
                return True
        return False

    def _record_scrap_time_based(self):
        """Record a scrap event with timestamp"""
        current_time = self.env.now
        self.scrap_events.append({'timestamp': current_time, 'scrap': 1})
        # Keep only recent events
        cutoff_time = current_time - (self.scrap_window_size * 10)
        self.scrap_events = [
            event for event in self.scrap_events
            if event['timestamp'] > cutoff_time
        ]

    def _get_moving_window_scrap(self, window_end_time=None):
        """Compute scrapped parts in a moving window ending at specified time"""
        if window_end_time is None:
            window_end_time = self.env.now
        window_start_time = window_end_time - self.scrap_window_size
        window_events = [
            event for event in self.scrap_events
            if window_start_time <= event['timestamp'] <= window_end_time
        ]
        total_scrap = sum(event['scrap'] for event in window_events)
        scrap_rate = total_scrap / self.scrap_window_size
        return {
            'scrap_in_window': total_scrap,
            'scrap_rate': scrap_rate,
            'window_start': window_start_time,
            'window_end': window_end_time
        }

    def _get_moving_average_scrap(self, num_windows=None):
        """Compute moving average of scrapped parts over multiple windows"""
        if num_windows is None:
            num_windows = self.scrap_moving_window_lookback
        current_time = self.env.now
        window_rates = []
        for i in range(num_windows):
            window_end = current_time - (i * self.scrap_window_size)
            if window_end >= self.scrap_window_size:
                window_data = self._get_moving_window_scrap(window_end)
                window_rates.append(window_data['scrap_rate'])
        if window_rates:
            return sum(window_rates) / len(window_rates)
        else:
            return 0.0

    def _update_scrap_metrics(self):
        """Update scrap metrics in state"""
        if not hasattr(self, 'state'):
            return
            
        current_window = self._get_moving_window_scrap()
        
        # Update scrap rate (scrap per time unit)
        if 'scrap_rate' in self.state.names:
            self.state['scrap_rate'].update(current_window['scrap_rate'])
            
        # Update moving average scrap rate
        if 'avg_scrap_last_5_windows' in self.state.names:
            avg_rate = self._get_moving_average_scrap(5)
            self.state['avg_scrap_last_5_windows'].update(avg_rate)
            
        # Optional: Update window scrap count if you want to keep this metric
        if 'scrap_in_window' in self.state.names:
            self.state['scrap_in_window'].update(current_window['scrap_in_window'])

    def _update_throughput_metrics(self):
        """Override to include scrap metrics"""
        super()._update_throughput_metrics()  # Call parent method
        self._update_scrap_metrics()  # Add scrap metrics
    
    def _draw_info(self, screen):
        self._draw_n_workers(screen)

    def _get_edge_features_to_pool(self):
        features = []
        # throughput rate, buffer_in/buffer_out fill rates
        throughput_rate = self.state['throughput_rate'].value if 'throughput_rate' in self.state.names else 0.0
        in_buffer_fill_rate = self.buffer_in.__self__.get_fillstate()
        out_buffer_fill_rate = self.buffer_out.__self__.get_fillstate()
        features.extend([throughput_rate, in_buffer_fill_rate, out_buffer_fill_rate])
        return np.array(features, dtype=np.float32)

    def run(self):

        while True:
            if self.is_on():

                yield self.env.process(self.request_workers())
                if 'n_workers_proportion' in self.state.names:
                    self.state['n_workers_proportion'].update(self.n_workers_proportional)
                self.state['n_workers'].update(self.n_workers)
                # Wait to get part from buffer_in
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                carrier = yield self.env.process(self.buffer_in())

                # Update current_carrier and count parts of carrier
                self.state['carrier'].update(carrier.name)

                # Run until carrier with components each having a valid assembly condition is
                # received
                while True:
                    # Wait to get component
                    carrier_component = yield self.env.process(self.buffer_component())
                    self.state['carrier_component'].update(carrier_component.name)

                    # Check component
                    if self._has_invalid_components_on_carrier(carrier_component):
                        yield self.env.process(self.set_to_error())
                        self._record_utilization_time_based('failing')
                        yield self.env.timeout(self.NOK_part_error_time)
                        self.state['n_scrap_parts'].increment()
                        # record scrap event
                        self._record_scrap_time_based()
                        # Update scrap rate metrics immediately after scrap event
                        if self.use_rates:
                            self._update_scrap_metrics()
                        # send carrier back
                        if hasattr(self, 'buffer_return'):
                            carrier_component.parts.clear()
                            yield self.env.process(self.buffer_return(carrier_component))
                        yield self.env.process(self.set_to_waiting())
                        self._record_utilization_time_based('waiting')
                        continue

                    else:
                        # All components are valid, proceed with assembly
                        break

                # Process components
                yield self.env.process(self.set_to_work())
                self._record_utilization_time_based('working')
                processing_time = self._sample_exp_time(
                    time=self.processing_time + carrier.get_additional_processing_time(self.name),
                    scale=self.processing_std,
                )
                yield self.env.timeout(processing_time)
                self.state['processing_time'].update(processing_time)

                # Add this line after updating processing_time
                if self.use_normalization:
                    self._update_normalized_states()
                for component in carrier_component:
                    carrier.assemble(component)

                # Release workers
                self.release_workers()
                
                # Record time-based throughput
                self._record_throughput_time_based(carrier)
                
                # Update moving averages
                self._update_throughput_metrics()

                # Place carrier on buffer_out
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                yield self.env.process(self.buffer_out(carrier))
                self.state['carrier'].update(None)

                # send component carrier back
                if hasattr(self, 'buffer_return'):
                    yield self.env.process(self.buffer_return(carrier_component))

                self.state['carrier_component'].update(None)

            else:
                yield self.turn_off()

class Process(Station):
    '''
    Process stations take a carrier as input, process the carrier, and push it onto buffer_out
    Args:
        processing_std: Standard deviation of the processing time
        rework_probability: Probability of a carrier to be reworked (takes 2x the time)
        position (tuple): X and Y position in visualization
    '''

    def __init__(
        self,
        name,
        buffer_in=None,
        buffer_out=None,
        processing_time=5,
        position=None,
        processing_std=None,
        rework_probability=0,
        worker_pool=None,
        min_processing_time = 0,
        actionable_processing_time=False,
        use_rates=False,
        use_normalization=False,
    ):

        super().__init__(
            name=name,
            position=position,
            processing_time=processing_time,
            processing_std=processing_std,
            rework_probability=rework_probability,
            worker_pool=worker_pool,
            use_rates=use_rates,
            use_normalization=use_normalization
        )

        self.min_processing_time = min_processing_time
        self.actionable_processing_time = actionable_processing_time
        if buffer_in is not None:
            self._connect_to_input(buffer_in)

        if buffer_out is not None:
            self._connect_to_output(buffer_out)

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        3 observables
        - on                    (non-observable)
        - mode                  (observable)
        - carrier               (non-observable)
        - n_workers             (observable)
        - processing_time       (observable)
        
        rate-wise states:
        5 observables
        - on                    (non-observable)
        - mode                  (observable)
        - carrier               (non-observable)
        - n_workers             (non-observable)
        - processing_time       (observable)
        ------------------------------------------------
        - n_workers_proportion  (observable)
        - throughput_rate       (observable)
        - utilization_rate      (observable)

        """
        if self.use_rates and self.use_normalization==False:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                TokenState(name='carrier', is_observable=False),
                NumericState('processing_time', is_actionable=False, is_observable=True, vmin=0),
                CountState('n_workers', is_actionable=False, is_observable=False, vmin=0),
                # rate specific metrics
                NumericState('n_workers_proportion', is_actionable=False, is_observable=True, vmin=0),
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        elif self.use_normalization and self.use_rates:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing'], is_observable=False),
                NumericState('norm_mode', is_actionable=False, is_observable=True, vmin=0),
                TokenState(name='carrier', is_observable=False),
                NumericState('processing_time', is_actionable=False, is_observable=False, vmin=0),
                NumericState('norm_processing_time', is_actionable=False, is_observable=True, vmin=0),
                CountState('n_workers', is_actionable=False, is_observable=False, vmin=0),
                # rate specific metrics
                NumericState('n_workers_proportion', is_actionable=False, is_observable=True, vmin=0),
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        else:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                TokenState(name='carrier', is_observable=False),
                NumericState('processing_time', is_actionable=False, is_observable=True, vmin=0),
                CountState('n_workers', is_actionable=False, is_observable=True, vmin=0),
            )
        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)
        self.state['processing_time'].update(self.processing_time)
        self.state['n_workers'].update(self.n_workers)
        if self.use_rates:
            self.state['n_workers_proportion'].update(self.n_workers_proportional)
            self.state['throughput_rate'].update(0.0)
            self.state['utilization_rate'].update(0.0)
            if self.use_normalization:
                self.state['norm_mode'].update(self.state['mode'].value)
                self.state['norm_processing_time'].update(self.state['processing_time'].value)

    def _draw_info(self, screen):
        self._draw_n_workers(screen)

    def run(self):

        while True:
            if self.is_on():
                yield self.env.process(self.request_workers())
                if 'n_workers_proportion' in self.state.names:
                    self.state['n_workers_proportion'].update(self.n_workers_proportional)
                self.state['n_workers'].update(self.n_workers)
                # Wait to get part from buffer_in
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                carrier = yield self.env.process(self.buffer_in())
                self.state['carrier'].update(carrier.name)

                yield self.env.process(self.set_to_work())
                self._record_utilization_time_based('working')
                processing_time = self._sample_exp_time(
                    time=self.processing_time + carrier.get_additional_processing_time(self.name),
                    scale=self.processing_std,
                    rework_probability=self.rework_probability,
                )
                ## just testing
                processing_time = int(min(99, processing_time))
                yield self.env.timeout(processing_time)
                self.state['processing_time'].update(processing_time)
                if self.use_normalization:
                    self._update_normalized_states()
                # Release workers
                self.release_workers()

                # Record time-based throughput
                self._record_throughput_time_based(carrier)
                
                # Update moving averages
                self._update_throughput_metrics()

                # Wait to place carrier to buffer_out
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                yield self.env.process(self.buffer_out(carrier))
                self.state['carrier'].update(None)

            else:
                yield self.turn_off()


class Source(Station):
    '''
    Source station generating parts on carriers.

    The Source takes carriers from buffer_in, creates a part, places that part
    onto the carrier, and pushes the carrier onto the buffer_out.
    If unlimited_carriers is True, no buffer_in is needed and no magazine.

    Args:
        name (str): Name of the Cell
        carrier_specs (dict): Nested dict. Top level descripes carrier types, each consists of a
            dict specifying different parts setup on the carrier at the source. The part level
            specifies how the part behaves at future processes along the layout. For instance a spec
            `{'C': {'Part1': {'Process1': {'assembly_condition': 5}, 'Process2': {'extra_processing_time': 10}}}}` 
            tells that the produced carrier has one part `Part1` that has to fullfill an assembly condition of `5` 
            at station `Process1` and gets an additional processing time of `10` at `Process2`.
        buffer_in (lineflow.simulation.connectors.Buffer, optional): Buffer in
        buffer_out (obj): Buffer out
        processing_time (float): Time it takes to put part on carrier (carrier needs to be
            available)
        processing_std (float): Standard deviation of processing time
        waiting_time (float): Time to wait between pushing a carrier out and taking the next one
        position (tuple): X and Y position in visualization
        unlimited_carriers (bool): If source has the ability to create unlimited carriers
        carrier_capacity (int): Defines how many parts can be assembled on a carrier. If set to
            default (infinity) or > 15, carrier will be visualized with one part.
        carrier_min_creation (int): Minimum number of carriers of same spec created subsequentially
        carrier_max_creation (int): Maximum number of carriers of same spec created subsequentially

    '''
    def __init__(
        self,
        name,
        buffer_in=None,
        buffer_out=None,
        processing_time=2,
        processing_std=None,
        waiting_time=0,
        waiting_time_step=0.5,
        position=None,
        actionable_magazin=True,
        actionable_waiting_time=True,
        unlimited_carriers=False,
        carrier_capacity=np.inf,
        carrier_specs=None,
        carrier_min_creation=1,
        carrier_max_creation=None,
        use_rates=False,
        use_normalization=False
    ):
        super().__init__(
            name=name,
            position=position,
            processing_time=processing_time,
            processing_std=processing_std,
            use_rates=use_rates,
            use_normalization=use_normalization
        )
        self._assert_init_args(unlimited_carriers, carrier_capacity, buffer_in)

        if buffer_in is not None:
            self._connect_to_input(buffer_in)
        if buffer_out is not None:
            self._connect_to_output(buffer_out)

        self.buffer_in_object = buffer_in
        self.unlimited_carriers = unlimited_carriers
        self.carrier_capacity = carrier_capacity
        self.waiting_time_step = waiting_time_step

        self.actionable_magazin = actionable_magazin
        self.actionable_waiting_time = actionable_waiting_time

        if carrier_specs is None:
            carrier_specs = {"carrier": {"part": {}}}
        self.carrier_specs = carrier_specs

        self.unlimited_carriers = unlimited_carriers
        self.carrier_capacity = carrier_capacity
        self.carrier_id = 1
        self.carrier_min_creation = carrier_min_creation
        self.carrier_max_creation = carrier_max_creation if carrier_max_creation is not None else 2*carrier_min_creation

        self._carrier_counter = 0

        self.init_waiting_time = waiting_time

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        3 observables
        - on                    (non-observable)
        - mode                  (observable)
        - waiting_time          (observable)
        - carrier               (non-observable)
        - part                  (non-observable)
        - carrier_spec          (observable)
        
        rate-wise states:
        5 observables
        - on                    (non-observable)
        - mode                  (observable)
        - waiting_time          (observable)
        - carrier               (non-observable)
        - part                  (non-observable)
        - carrier_spec          (observable)
        ------------------------------------------------
        - throughput_rate       (observable)
        - utilization_rate      (observable)

        """
        if self.use_rates and self.use_normalization == False:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                DiscreteState(
                    name='waiting_time', 
                    categories=np.arange(0, 100, self.waiting_time_step), 
                    is_actionable=self.actionable_waiting_time,
                ),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='part', is_observable=False),
                DiscreteState(
                    name='carrier_spec', 
                    categories=list(self.carrier_specs.keys()), 
                    is_actionable=False,
                    is_observable=True,
                ),
                # rate specific metrics
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        elif self.use_rates and self.use_normalization:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing'], is_observable=False),
                NumericState('norm_mode', is_actionable=False, is_observable=True, vmin=0),
                DiscreteState(
                    name='waiting_time', 
                    categories=np.arange(0, 100, self.waiting_time_step), 
                    is_actionable=self.actionable_waiting_time,
                    is_observable=False
                ),
                NumericState('norm_waiting_time', is_actionable=False, is_observable=True, vmin=0),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='part', is_observable=False),
                DiscreteState(
                    name='carrier_spec', 
                    categories=list(self.carrier_specs.keys()), 
                    is_actionable=False,
                    is_observable=True,
                ),
                # rate specific metrics
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        else:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                DiscreteState(
                    name='waiting_time', 
                    categories=np.arange(0, 100, self.waiting_time_step), 
                    is_actionable=self.actionable_waiting_time,
                ),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='part', is_observable=False),
                DiscreteState(
                    name='carrier_spec', 
                    categories=list(self.carrier_specs.keys()), 
                    is_actionable=False,
                    is_observable=True,
                ),
            )
        self.state['waiting_time'].update(self.init_waiting_time)
        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)
        self.state['part'].update(None)
        self.state['carrier_spec'].update(list(self.carrier_specs.keys())[0])
        if self.use_rates:
            self.state['throughput_rate'].update(0.0)
            self.state['utilization_rate'].update(0.0)
            if self.use_normalization:
                self.state['norm_mode'].update(0.0)
                self.state['norm_waiting_time'].update(0.0)

    def apply(self, actions):
        """Override apply to update normalized states when discrete states change"""
        super().apply(actions)
        
        # Update normalized states after applying actions
        if self.use_normalization:
            self._update_normalized_states()

    def _assert_init_args(self, unlimited_carriers, carrier_capacity, buffer_in):
        if unlimited_carriers:
            if carrier_capacity > 15:
                warnings.warn(
                    "If carrier_capacity > 15, visualization of parts"
                    "on carriers is restriced and carrier will be visualized with one part")
            if not isinstance(carrier_capacity, int) and carrier_capacity != np.inf:
                raise AttributeError("Type of carrier capacity must be int or np.inf")

    def create_carrier(self):

        if self._carrier_counter == 0:
            carrier_spec = self.random.choice(list(self.carrier_specs.keys()))
            self.state['carrier_spec'].update(carrier_spec)
            self._carrier_counter = self.random.randint(
                self.carrier_min_creation, 
                self.carrier_max_creation + 1,
            )

        carrier_spec = self.state['carrier_spec'].to_str()

        name = f'{self.name}_{carrier_spec}_{self.carrier_id}'
        carrier = Carrier(
            self.env, 
            name=name, 
            capacity=self.carrier_capacity, 
            part_specs=self.carrier_specs[carrier_spec],
        )
        self.carrier_id += 1
        self._carrier_counter -= 1

        return carrier

    def create_parts(self, carrier):
        """
        Creates the parts based on the part_specs attribute
        For each dict in the part_specs list one part is created
        """

        parts = []
        for part_id, (part_name, part_spec) in enumerate(carrier.part_specs.items()):
            part = Part(
                env=self.env,
                name=f"{carrier.name}_{part_name}_{part_id}",
                specs=part_spec,
            )
            part.create(self.position)
            parts.append(part)
        return parts

    def assemble_parts_on_carrier(self, carrier, parts):
        """
        Put parts onto carrier
        """
        for part in parts:
            carrier.assemble(part)

    def assemble_carrier(self, carrier):

        parts = self.create_parts(carrier)
        self.state['part'].update(parts[0].name)

        processing_time = self._sample_exp_time(
            time=self.processing_time,
            scale=self.processing_std,
        )
        self.state['carrier'].update(carrier.name)

        yield self.env.timeout(processing_time)
        self.assemble_parts_on_carrier(carrier, parts)

        return carrier

    def wait(self):

        waiting_time = self.state['waiting_time'].to_str()
        
        # Update normalized waiting time after any programmatic changes
        if self.use_normalization:
            self._update_normalized_states()

        if waiting_time > 0:
            yield self.env.process(self.set_to_waiting())
            self._record_utilization_time_based('waiting')
            yield self.env.timeout(waiting_time)

    def run(self):

        while True:
            if self.is_on():
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                yield self.env.process(self.wait())

                if self.unlimited_carriers:
                    carrier = self.create_carrier()
                else:
                    carrier = yield self.env.process(self.buffer_in())

                yield self.env.process(self.set_to_work())
                self._record_utilization_time_based('working')
                carrier = yield self.env.process(self.assemble_carrier(carrier))

                # Record time-based throughput
                self._record_throughput_time_based(carrier)
                
                # Update moving averages
                self._update_throughput_metrics()

                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                yield self.env.process(self.buffer_out(carrier))
                self.state['part'].update(None)
                self.state['carrier'].update(None)

            else:
                yield self.turn_off()


class Sink(Station):
    """
    The Sink takes carriers from buffer_in. It removes the parts of the carrier and either
    destroys it or puts them to buffer_out if one exists.

    Args:
        processing_std (float): The standard deviation of the processing time.
        position (tuple): X and Y position in the visualization.
    """
    def __init__(
        self,
        name,
        buffer_in=None,
        buffer_out=None,
        processing_time=2,
        processing_std=None,
        position=None,
        use_rates=False,
        use_normalization=False
    ):
        super().__init__(
            name=name,
            processing_time=processing_time,
            processing_std=processing_std,
            position=position,
            use_rates=use_rates,
            use_normalization=use_normalization
        )

        if buffer_in is not None:
            self._connect_to_input(buffer_in)
        if buffer_out is not None:
            self._connect_to_output(buffer_out)

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        1 observables
        - on                    (non-observable)
        - mode                  (observable)
        - n_parts_produced      (non-observable)
        - carrier               (non-observable)
        
        rate-wise states:
        3 observables
        - on                    (non-observable)
        - mode                  (observable)
        - n_parts_produced      (non-observable)
        - carrier               (non-observable)
        ------------------------------------------------
        - throughput_rate       (observable)
        - utilization_rate      (observable)

        """
        if self.use_rates and self.use_normalization == False:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                CountState('n_parts_produced', is_actionable=False, is_observable=False),
                TokenState(name='carrier', is_observable=False),
                # rate
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0)
            )
        elif self.use_rates and self.use_normalization:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing'], is_observable=False),
                NumericState('norm_mode', is_actionable=False, is_observable=True, vmin=0),
                CountState('n_parts_produced', is_actionable=False, is_observable=False),
                TokenState(name='carrier', is_observable=False),
                # rate
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0)
            )
        else:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                CountState('n_parts_produced', is_actionable=False, is_observable=False),
                TokenState(name='carrier', is_observable=False),
            )

        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['n_parts_produced'].update(0)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)

        if self.use_rates:
            self.state['throughput_rate'].update(0.0)
            self.state['utilization_rate'].update(0.0)
            if self.use_normalization:
                self.state['norm_mode'].update(0.0)

    def remove(self, carrier):

        processing_time = self._sample_exp_time(
            time=self.processing_time,
            scale=self.processing_std,
        )
        yield self.env.timeout(processing_time)
        self.state['n_parts_produced'].increment()

        if hasattr(self, 'buffer_out'):
            yield self.env.process(self.set_to_waiting())
            self._record_utilization_time_based('waiting')
            carrier.parts.clear()
            yield self.env.process(self.buffer_out(carrier))

    def run(self):

        while True:
            if self.is_on():
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                carrier = yield self.env.process(self.buffer_in())
                yield self.env.process(self.set_to_work())
                self._record_utilization_time_based('working')
                self.state['carrier'].update(carrier.name)

                # Wait to place carrier to buffer_out
                yield self.env.process(self.remove(carrier))

                # Record time-based throughput
                self._record_throughput_time_based(carrier)
                
                # Update moving averages
                self._update_throughput_metrics()

                self.state['carrier'].update(None)

            else:
                yield self.turn_off()


class Switch(Station):
    """
    A Switch distributes carriers onto buffer outs. In and out buffers can be provided to
    the constructor but can also be added to a switch using the `connect_to_input` and `connect_to_output`
    methods.

    Args:
        buffers_in (list): A list of buffers that lead into the Switch.
        buffers_out (list): A list of buffers that lead away from the Switch.
        position (tuple): X and Y position in the visualization.
        alternate (bool): If True, the Switch switches between the buffers_out; else, only one buffer_out is used.
    """

    def __init__(
        self,
        name,
        buffers_in=None,
        buffers_out=None,
        position=None,
        processing_time=5,
        alternate=False,
        use_rates=False,
        use_normalization=False
    ):
        super().__init__(
            name=name,
            position=position,
            processing_time=processing_time,
            # We assume switches do not have variation here
            processing_std=0,
            use_rates=use_rates,
            use_normalization=use_normalization,
        )

        # time it takes for a model to change buffer_in or buffer_out
        self.readjustment_time = 10

        self.buffer_in = []
        self.buffer_out = []

        if buffers_in is not None:
            for buffer in buffers_in:
                self._connect_to_input(buffer)

        if buffers_out is not None:
            for buffer in buffers_out:
                self._connect_to_output(buffer)

        self.alternate = alternate

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        3 observables
        - on                    (non-observable)
        - mode                  (observable)
        - index_buffer_in       (observable)
        - index_buffer_out      (observable)

        rate-wise states:
        9 observables
        - on                    (non-observable)
        - mode                  (observable)
        - index_buffer_in       (observable)
        - index_buffer_out      (observable)
        ------------------------------------------------
        - avg_fill_up_stream        (observable)
        - avg_fill_down_stream      (observable)
        - current_buffer_in_fill    (observable)
        - current_buffer_out_fill   (observable)
        - throughput_rate           (observable)
        - utilization_rate          (observable)

        """
        if self.use_rates and self.use_normalization == False:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                DiscreteState(
                    name='index_buffer_in',
                    categories=list(range(self.n_buffers_in)),
                    is_actionable=not self.alternate and self.n_buffers_in > 1
                ),
                DiscreteState(
                    name='index_buffer_out',
                    categories=list(range(self.n_buffers_out)),
                    is_actionable=not self.alternate and self.n_buffers_out > 1),
                TokenState(name='carrier', is_observable=False),
                # rate specific metrics
                NumericState('avg_fill_up_stream', is_actionable=False, is_observable=True,vmin=0),
                NumericState('avg_fill_down_stream', is_actionable=False, is_observable=True,vmin=0),
                NumericState('current_buffer_in_fill', is_actionable=False, is_observable=True, vmin=0, vmax=1),
                NumericState('current_buffer_out_fill', is_actionable=False, is_observable=True, vmin=0, vmax=1),
                NumericState('throughput_rate', is_actionable=False, is_observable=True,vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True,vmin=0),
            )
        elif self.use_normalization and self.use_rates:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                NumericState('norm_mode', is_actionable=False, is_observable=True, vmin=0),
                DiscreteState(
                    name='index_buffer_in',
                    categories=list(range(self.n_buffers_in)),
                    is_actionable=not self.alternate and self.n_buffers_in > 1,
                    is_observable=False
                ),
                DiscreteState(
                    name='index_buffer_out',
                    categories=list(range(self.n_buffers_out)),
                    is_actionable=not self.alternate and self.n_buffers_out > 1,
                    is_observable=False
                ),
                NumericState(
                    name='normalized_index_buffer_in',
                    is_actionable=False,
                    is_observable=True,
                    vmin=0
                ),
                NumericState(
                    name='normalized_index_buffer_out',
                    is_actionable=False,
                    is_observable=True,
                    vmin=0
                ),
                TokenState(name='carrier', is_observable=False),
                # rate specific metrics
                NumericState('avg_fill_up_stream', is_actionable=False, is_observable=True,vmin=0),
                NumericState('avg_fill_down_stream', is_actionable=False, is_observable=True,vmin=0),
                NumericState('current_buffer_in_fill', is_actionable=False, is_observable=True, vmin=0, vmax=1),
                NumericState('current_buffer_out_fill', is_actionable=False, is_observable=True, vmin=0, vmax=1),
                NumericState('throughput_rate', is_actionable=False, is_observable=True,vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True,vmin=0),
            )
        else:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                DiscreteState(
                    name='index_buffer_in',
                    categories=list(range(self.n_buffers_in)),
                    is_actionable=not self.alternate and self.n_buffers_in > 1
                ),
                DiscreteState(
                    name='index_buffer_out',
                    categories=list(range(self.n_buffers_out)),
                    is_actionable=not self.alternate and self.n_buffers_out > 1
                ),
                TokenState(name='carrier', is_observable=False),
            )
        self.state['index_buffer_in'].update(0)
        self.state['index_buffer_out'].update(0)
        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)
        if self.use_rates:
            self.state['avg_fill_up_stream'].update(0.0)
            self.state['avg_fill_down_stream'].update(0.0)
            self.state['current_buffer_in_fill'].update(0.0)
            self.state['current_buffer_out_fill'].update(0.0)
            self.state['throughput_rate'].update(0.0)
            self.state['utilization_rate'].update(0.0)
            if self.use_normalization:
                self.state['normalized_index_buffer_in'].update(0.0)
                self.state['normalized_index_buffer_out'].update(0.0)
    
    def _normalize_buffer_index(self, index, num_buffers):
        """Normalize buffer index to [0, 1] range"""
        if num_buffers <= 1:
            return 0.0
        return index / (num_buffers - 1)

    def _normalize_buffer_index_new(self, index):
        """Normalize buffer index to [0, 1] range
        with the maximum number of buffers set to 5"""
        max_buffers = 5
        return index / (max_buffers - 1)

    def _update_normalized_indices(self):
        """Update normalized indices based on current discrete indices"""
        if not self.use_normalization:
            return
            
        # Get current discrete indices
        current_in_index = self.state['index_buffer_in'].value
        current_out_index = self.state['index_buffer_out'].value
        
        # Normalize indices
        # normalized_in = self._normalize_buffer_index(current_in_index, self.n_buffers_in)
        # normalized_out = self._normalize_buffer_index(current_out_index, self.n_buffers_out)

        normalized_in = self._normalize_buffer_index_new(current_in_index)
        normalized_out = self._normalize_buffer_index_new(current_out_index)

        # Update normalized states
        self.state['normalized_index_buffer_in'].update(normalized_in)
        self.state['normalized_index_buffer_out'].update(normalized_out)

    def _alternate_indices(self):
        self.state['index_buffer_in'].set_next()
        self.state['index_buffer_out'].set_next()
        
        # Update normalized indices when discrete indices change
        if self.use_normalization:
            self._update_normalized_indices()

    def apply(self, actions):
        """Override apply to update normalized indices when discrete indices change"""
        super().apply(actions)
        
        # Update normalized indices after applying actions
        if self.use_normalization:
            self._update_normalized_indices()
    @property
    def n_buffers_in(self):
        return len(self.buffer_in)

    @property
    def n_buffers_out(self):
        return len(self.buffer_out)

    def _get_buffer_in_position(self):
        return self.buffer_in[
            self.state['index_buffer_in'].value
        ].__self__._positions_slots[-1]

    def _get_buffer_out_position(self):
        return self.buffer_out[
            self.state['index_buffer_out'].value
        ].__self__._positions_slots[0]

    def _draw_info(self, screen):
        pos_buffer_in = self._get_buffer_in_position()
        pos_buffer_out = self._get_buffer_out_position()

        pos_in = pos_buffer_in + 0.5*(self.position - pos_buffer_in)
        pos_out = pos_buffer_out + 0.5*(self.position - pos_buffer_out)

        pygame.draw.circle(screen, 'gray', self.position, 6)
        for pos in [pos_in, pos_out]:
            pygame.draw.line(screen, "gray", self.position, pos, width=5)

    def _connect_to_input(self, buffer):
        self.buffer_in.append(buffer.connect_to_output(self))

    def _connect_to_output(self, buffer):
        self.buffer_out.append(buffer.connect_to_input(self))

    def _alternate_indices(self):
        self.state['index_buffer_in'].set_next()
        self.state['index_buffer_out'].set_next()

    def look_in(self):
        """
        Checks if part at current buffer_in is available
        """
        buffer_in = self.buffer_in[self.state['index_buffer_in'].value].__self__
        while buffer_in.get_fillstate() == 0:
            yield self.env.timeout(1)
            buffer_in = self.buffer_in[self.state['index_buffer_in'].value].__self__
        return buffer_in

    def look_out(self):
        """
        Checks if space at current buffer_out is available
        """
        buffer_out = self.buffer_out[self.state['index_buffer_out'].value].__self__

        while buffer_out.get_fillstate() == 1:
            yield self.env.timeout(1)
            # check if buffer out changed
            buffer_out = self.buffer_out[self.state['index_buffer_out'].value].__self__
        return buffer_out

    def get(self):
        while True:
            yield self.env.process(self.set_to_waiting())
            self._record_utilization_time_based('waiting')
            buffer_in = yield self.env.process(self.look_in())
            self.getting_process = None
            yield self.env.process(self.set_to_work())
            self._record_utilization_time_based('working')
            carrier = yield self.env.process(
                buffer_in.get()
            )
            self.state['carrier'].update(carrier.name)
            return carrier

    def put(self, carrier):
        while True:
            yield self.env.process(self.set_to_waiting())
            self._record_utilization_time_based('waiting')
            buffer_out = yield self.env.process(self.look_out())
            yield self.env.process(buffer_out.put(carrier))
            self.state['carrier'].update(None)
            return

    def get_input_buffer_fill_rates(self):
        """Get fill rates of all input buffers"""
        fill_rates = []
        for buffer_method in self.buffer_in:
            buffer_obj = buffer_method.__self__  # Get the actual buffer object
            fill_rate = buffer_obj.get_fillstate()  # This returns fill percentage (0.0 to 1.0)
            fill_rates.append(fill_rate)
        return fill_rates

    def get_current_input_buffer_fill_rate(self):
        """Get fill rate of currently selected input buffer"""
        current_buffer_idx = self.state['index_buffer_in'].value
        buffer_obj = self.buffer_in[current_buffer_idx].__self__
        return buffer_obj.get_fillstate()

    def get_output_buffer_fill_rates(self):
        """Get fill rates of all output buffers"""
        fill_rates = []
        for buffer_method in self.buffer_out:
            buffer_obj = buffer_method.__self__  # Get the actual buffer object
            fill_rate = buffer_obj.get_fillstate()  # This returns fill percentage (0.0 to 1.0)
            fill_rates.append(fill_rate)
        return fill_rates

    def get_current_output_buffer_fill_rate(self):
        """Get fill rate of currently selected output buffer"""
        current_buffer_idx = self.state['index_buffer_out'].value
        buffer_obj = self.buffer_out[current_buffer_idx].__self__
        return buffer_obj.get_fillstate()
    
    def _update_buffer_fill_metrics(self):
        """Update buffer fill rate metrics in state"""
        if not hasattr(self, 'state') or not self.use_rates:
            return

        # Update upstream (input) buffer fill rates
        input_fill_rates = self.get_input_buffer_fill_rates()
        if input_fill_rates:
            avg_upstream_fill = sum(input_fill_rates) / len(input_fill_rates)
            self.state['avg_fill_up_stream'].update(avg_upstream_fill)
            
            current_input_fill = input_fill_rates[self.state['index_buffer_in'].value]
            self.state['current_buffer_in_fill'].update(current_input_fill)

        # Update downstream (output) buffer fill rates
        output_fill_rates = self.get_output_buffer_fill_rates()
        if output_fill_rates:
            avg_downstream_fill = sum(output_fill_rates) / len(output_fill_rates)
            self.state['avg_fill_down_stream'].update(avg_downstream_fill)

            current_output_fill = output_fill_rates[self.state['index_buffer_out'].value]
            self.state['current_buffer_out_fill'].update(current_output_fill)
    
    def _update_throughput_metrics(self):
        """Override to include buffer fill metrics"""
        super()._update_throughput_metrics()  # Call parent method
        self._update_buffer_fill_metrics()  # Add buffer fill metrics

    def run(self):
        while True:
            if self.is_on():
                # Get carrier
                carrier = yield self.env.process(self.get())

                # Process
                yield self.env.process(self.set_to_work())
                self._record_utilization_time_based('working')
                yield self.env.timeout(self.processing_time)

                yield self.env.process(self.put(carrier))

                # Record time-based throughput
                self._record_throughput_time_based(carrier)
                
                # Update moving averages
                self._update_throughput_metrics()

                if self.alternate:
                    self._alternate_indices()

            else:
                yield self.turn_off()


class Magazine(Station):
    '''
    Magazine station manages carriers.

    The Magazine gets carriers from buffer_in and stores them in the
    magazine. Afterwards it takes a carrier from its magazine and pushes the
    carrier to buffer_out.
    If unlimited_carriers is True no buffer_in is needed.

    Args:
        unlimited_carriers (bool): If True, the Magazine will have an unlimited amount of carriers available
        carriers_in_magazine (int): Number of carriers in the magazine
        carrier_getting_time (float): Time to get a carrier from the magazine
        actionable_magazine (bool): If True, carriers in the magazine is in an actionable state
        carrier_capacity (int): Defines how many parts can be assembled on a carrier. If set to default (infinity) or > 15, carrier will be visualized with one part.
    '''
    def __init__(
        self,
        name,
        buffer_in=None,
        buffer_out=None,
        position=None,
        unlimited_carriers=True,
        carrier_capacity=np.inf,
        actionable_magazine=True,
        carrier_getting_time=2,
        carriers_in_magazine=0,
        carrier_specs=None,
        carrier_min_creation=1,
        carrier_max_creation=None,
        use_rates=False,
        use_normalization=False,
    ):
        super().__init__(
            name=name,
            position=position,
            use_rates=use_rates,
            use_normalization=use_normalization,
        )
        self._assert_init_args(buffer_in, unlimited_carriers, carriers_in_magazine, carrier_capacity)

        if buffer_in is not None:
            self._connect_to_input(buffer_in)
        if buffer_out is not None:
            self._connect_to_output(buffer_out)

        self.actionable_magazine = actionable_magazine
        self.init_carriers_in_magazine = carriers_in_magazine
        self.carrier_getting_time = carrier_getting_time

        if carrier_specs is None:
            carrier_specs = {"carrier": {"part": {}}}
        self.carrier_specs = carrier_specs

        self.unlimited_carriers = unlimited_carriers
        self.carrier_capacity = carrier_capacity
        self.carrier_id = 1
        self.carrier_min_creation = carrier_min_creation
        self.carrier_max_creation = carrier_max_creation if carrier_max_creation is not None else 2*carrier_min_creation
        self._carrier_counter = 0
        # Add magazine state tracking
        self.magazine_state_events = []
        self.magazine_window_size = self.time_based_window_size
        self.magazine_moving_window_lookback = 5

    def init_state(self):
        """
        Initialize the state of the station.
        
        basic states (original):
        3 observables
        - on                    (non-observable)
        - mode                  (observable)
        - carriers_in_magazine  (observable)
        - carrier               (non-observable)
        - part                  (non-observable)

        rate-wise states:
        9 observables
        - on                    (non-observable)
        - mode                  (observable)
        - carriers_in_magazine  (observable)
        - carrier               (non-observable)
        - part                  (non-observable)
        ------------------------------------------------
        - magazine_utilization_rate (observable)
        - throughput_rate           (observable)
        - utilization_rate          (observable)

        """
        if self.use_rates and self.use_normalization == False:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                CountState('carriers_in_magazine', is_actionable=self.actionable_magazine, is_observable=False),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='part', is_observable=False),
                # rate-wise states
                NumericState('avg_carriers_in_magazine', is_actionable=False, is_observable=False, vmin=0),
                NumericState('magazine_utilization_rate', is_actionable=False, is_observable=True, vmin=0, vmax=1),
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        elif self.use_normalization and self.use_rates:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing'], is_observable=False),
                NumericState('norm_mode', is_actionable=False, is_observable=True, vmin=0),
                CountState('carriers_in_magazine', is_actionable=self.actionable_magazine, is_observable=False),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='part', is_observable=False),
                # rate-wise states
                NumericState('avg_carriers_in_magazine', is_actionable=False, is_observable=False, vmin=0),
                NumericState('magazine_utilization_rate', is_actionable=False, is_observable=True, vmin=0, vmax=1),
                NumericState('throughput_rate', is_actionable=False, is_observable=True, vmin=0),
                NumericState('utilization_rate', is_actionable=False, is_observable=True, vmin=0),
            )
        else:
            self.state = ObjectStates(
                DiscreteState('on', categories=[True, False], is_actionable=False, is_observable=False),
                DiscreteState('mode', categories=['working', 'waiting', 'failing']),
                CountState('carriers_in_magazine', is_actionable=self.actionable_magazine, is_observable=True),
                TokenState(name='carrier', is_observable=False),
                TokenState(name='part', is_observable=False),
            )
        

        self.state['carriers_in_magazine'].update(self.init_carriers_in_magazine)
        self.state['on'].update(True)
        self.state['mode'].update("waiting")
        self.state['carrier'].update(None)
        self.state['part'].update(None)
        if self.use_rates:
            self.state['avg_carriers_in_magazine'].update(float(self.init_carriers_in_magazine))
            self.state['magazine_utilization_rate'].update(1.0 if self.init_carriers_in_magazine > 0 else 0.0)
            self.state['throughput_rate'].update(0.0)
            self.state['utilization_rate'].update(0.0)


    def _assert_init_args(self, buffer_in, unlimited_carriers, carriers_in_magazine, carrier_capacity):
        if carrier_capacity > 15:
            warnings.warn("If carrier_capacity > 15, visualization of parts on carriers is restriced and carrier will be visualized with one part")

        if not isinstance(carrier_capacity, int) and carrier_capacity != np.inf:
            raise AttributeError("Type of carrier capacity must be int or np.inf")

        if not unlimited_carriers and carriers_in_magazine == 0:
            raise AttributeError(f"unlimited_carriers is {unlimited_carriers} and cell also has 0 carriers in magazine")

        if unlimited_carriers and carriers_in_magazine > 0:
            raise AttributeError(f"unlimited_carriers is {unlimited_carriers} and cell has more than 0 carriers in magazine")

        if buffer_in and unlimited_carriers:
                raise AttributeError(f"Only magazine or unlimited_carriers {unlimited_carriers} is required")

    def create_carrier(self):
        if self._carrier_counter == 0:
            self._current_carrier_spec = self.random.choice(list(self.carrier_specs.keys()))
            self._carrier_counter = self.random.randint(
                self.carrier_min_creation, 
                self.carrier_max_creation + 1,
            )

        name = f'{self.name}_{self._current_carrier_spec}_{self.carrier_id}'
        carrier = Carrier(
            self.env, 
            name=name, 
            capacity=self.carrier_capacity, 
            part_specs=self.carrier_specs[self._current_carrier_spec],
        )
        self.carrier_id += 1
        self._carrier_counter -= 1

        return carrier

    def _initial_fill_magazine(self, n_carriers):
        # attribute needs to be set here as env is not available in __init__()
        self.magazine = simpy.Store(self.env)
        for i in range(n_carriers):
            carrier = self.create_carrier()
            self.magazine.put(carrier)

    def get_carrier_from_magazine(self):
        yield self.env.process(self._update_magazine())
        yield self.env.timeout(self.carrier_getting_time)

        while True:
            yield self.env.process(self._update_magazine())
            yield self.env.process(self.set_to_work())
            self._record_utilization_time_based('working')
            if len(self.magazine.items) > 0:
                carrier = yield self.magazine.get()
                break
            else:
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                yield self.env.timeout(1)

        self.state['carriers_in_magazine'].decrement()
        if self.use_rates:
            self._record_magazine_state_time_based()
        return carrier

    def _buffer_in_to_magazine(self):
        while True:
            carrier = yield self.env.process(self.buffer_in())
            yield self.env.process(self.add_carrier_to_magazine(carrier))

    def add_carrier_to_magazine(self, carrier):
        yield self.magazine.put(carrier)
        self.state['carriers_in_magazine'].increment()
        # Record state change for rate calculation
        if self.use_rates:
            self._record_magazine_state_time_based()

    def _update_magazine(self):
        '''
        update the magazine according to state
        '''
        should = self.state['carriers_in_magazine'].value
        current = len(self.magazine.items)
        diff = should - current
        if diff > 0:
            for i in range(diff):
                carrier = self.create_carrier()
                self.magazine.put(carrier)

        if diff < 0:
            for i in range(abs(diff)):
                carrier = yield self.magazine.get()

    def _draw_info(self, screen):
        self._draw_n_carriers(screen)

    def get_carrier(self):
        # First check if Magazine is allowed to create unlimited carriers
        if self.unlimited_carriers:
            yield self.env.timeout(self.carrier_getting_time)
            carrier = self.create_carrier()

        # Second check magazine
        else:
            carrier = yield self.env.process(self.get_carrier_from_magazine())
        self.state["carrier"].update(carrier.name)
        return carrier

    def _record_magazine_state_time_based(self):
        """Record magazine state with timestamp"""
        current_time = self.env.now
        current_carriers = self.state['carriers_in_magazine'].value
        
        self.magazine_state_events.append({
            'timestamp': current_time,
            'carriers_count': current_carriers
        })
        
        # Keep only events within reasonable time (e.g., last 10 windows)
        cutoff_time = current_time - (self.magazine_window_size * 10)
        self.magazine_state_events = [
            event for event in self.magazine_state_events 
            if event['timestamp'] > cutoff_time
        ]

    def _get_moving_window_magazine_metrics(self, window_end_time=None):
        """Calculate magazine metrics for a moving window"""
        if window_end_time is None:
            window_end_time = self.env.now
        
        window_start_time = window_end_time - self.magazine_window_size
        
        # Find events within the window
        window_events = [
            event for event in self.magazine_state_events
            if window_start_time <= event['timestamp'] <= window_end_time
        ]
        
        if not window_events:
            current_carriers = self.state['carriers_in_magazine'].value
            return {
                'avg_carriers_in_magazine': current_carriers,
                'max_carriers_in_magazine': current_carriers,
                'min_carriers_in_magazine': current_carriers,
                'magazine_utilization_rate': current_carriers / self.init_carriers_in_magazine if self.init_carriers_in_magazine > 0 else 0.0,
                'magazine_fill_rate': current_carriers / self.init_carriers_in_magazine if self.init_carriers_in_magazine > 0 else 0.0
            }
        
        # Calculate statistics
        carrier_counts = [event['carriers_count'] for event in window_events]
        avg_carriers = sum(carrier_counts) / len(carrier_counts)
        max_carriers = max(carrier_counts)
        min_carriers = min(carrier_counts)
        
        # Magazine utilization rate (how full the magazine is on average)
        if hasattr(self, 'init_carriers_in_magazine') and self.init_carriers_in_magazine > 0:
            utilization_rate = avg_carriers / self.init_carriers_in_magazine
            fill_rate = avg_carriers / self.init_carriers_in_magazine
        else:
            # For unlimited magazines, use current max as reference
            utilization_rate = avg_carriers / max(1, max_carriers)
            fill_rate = avg_carriers / max(1, max_carriers)
        
        return {
            'avg_carriers_in_magazine': avg_carriers,
            'max_carriers_in_magazine': max_carriers,
            'min_carriers_in_magazine': min_carriers,
            'magazine_utilization_rate': utilization_rate,
            'magazine_fill_rate': fill_rate,
            'window_start': window_start_time,
            'window_end': window_end_time
        }

    def _get_moving_average_magazine_metrics(self, num_windows=None):
        """Calculate moving average of magazine metrics over multiple windows"""
        if num_windows is None:
            num_windows = self.magazine_moving_window_lookback
        
        current_time = self.env.now
        window_metrics = []
        
        for i in range(num_windows):
            window_end = current_time - (i * self.magazine_window_size)
            if window_end >= self.magazine_window_size:
                window_data = self._get_moving_window_magazine_metrics(window_end)
                window_metrics.append(window_data)
        
        if window_metrics:
            avg_carriers = sum(m['avg_carriers_in_magazine'] for m in window_metrics) / len(window_metrics)
            avg_utilization = sum(m['magazine_utilization_rate'] for m in window_metrics) / len(window_metrics)
            return {
                'avg_carriers_last_windows': avg_carriers,
                'avg_utilization_last_windows': avg_utilization
            }
        else:
            return {
                'avg_carriers_last_windows': 0.0,
                'avg_utilization_last_windows': 0.0
            }

    def _update_magazine_metrics(self):
        """Update magazine metrics in state"""
        if not hasattr(self, 'state') or not self.use_rates:
            return
        
        # Record current state
        self._record_magazine_state_time_based()
        
        # Get current window metrics
        current_window = self._get_moving_window_magazine_metrics()
        
        # Update state variables
        if 'avg_carriers_in_magazine' in self.state.names:
            self.state['avg_carriers_in_magazine'].update(current_window['avg_carriers_in_magazine'])
        
        if 'magazine_utilization_rate' in self.state.names:
            self.state['magazine_utilization_rate'].update(current_window['magazine_utilization_rate'])
        
        if 'magazine_fill_rate' in self.state.names:
            self.state['magazine_fill_rate'].update(current_window['magazine_fill_rate'])
        
        # Update moving averages
        moving_avg = self._get_moving_average_magazine_metrics(5)
        if 'avg_carriers_last_5_windows' in self.state.names:
            self.state['avg_carriers_last_5_windows'].update(moving_avg['avg_carriers_last_windows'])
        
        if 'avg_magazine_utilization_last_5_windows' in self.state.names:
            self.state['avg_magazine_utilization_last_5_windows'].update(moving_avg['avg_utilization_last_windows'])

    def _update_throughput_metrics(self):
        """Override to include magazine metrics"""
        super()._update_throughput_metrics()  # Call parent method
        self._update_magazine_metrics()  # Add magazine metrics

    def run(self):
        # Initially fill the magazine with carriers
        self._initial_fill_magazine(self.state['carriers_in_magazine'].value)

        if hasattr(self, 'buffer_in'):
            self.env.process(self._buffer_in_to_magazine())

        while True:
            if self.is_on():
                # Get carrier from Magazine
                yield self.env.process(self.set_to_work())
                self._record_utilization_time_based('working')
                carrier = yield self.env.process(self.get_carrier())

                # Record time-based throughput
                self._record_throughput_time_based(carrier)
                
                # Update moving averages
                self._update_throughput_metrics()
                
                # Wait to place carrier to buffer_out
                yield self.env.process(self.set_to_waiting())
                self._record_utilization_time_based('waiting')
                yield self.env.process(self.buffer_out(carrier))
                self.state['carrier'].update(None)
            else:
                yield self.turn_off()
##
'''
Assembly:
observation:['mode','n_scrap_parts','n_workers','processing_time']
actions: non-actionable station

Process:
observation:['mode','n_workers','processing_time']
actions: non-actionable station

Source:
observation:['mode','waiting_time','carrier_spec']
actions: ['waiting_time']

Sink:
observation:['mode']
actions: non-actionable station

Switch:
observation:['mode','index_buffer_in','index_buffer_out']
actions: ['index_buffer_in','index_buffer_out']

Magazine:
observation:['mode','carriers_in_magazine']
actions: ['carriers_in_magazine']

WorkerPool:
observation:['average throughput of connected stations']
actions: ['index_station']*n_workers

'''