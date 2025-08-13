from lineflow.simulation import (
    Buffer,
    Source,
    Sink,
    Line,
    Process,
    Magazine,
)
import matplotlib.pyplot as plt
import numpy as np

class WaterLine(Line): 
    def build(self):

        # Configure a simple line
        # buffer_1 = Buffer('Buffer1', capacity=50, transition_time=5)
        # buffer_2 = Buffer('Buffer2', capacity=50, transition_time=5)
        # buffer_3 = Buffer('Buffer3', capacity=50, transition_time=5)
        # buffer_4 = Buffer('Buffer4', capacity=50, transition_time=5)
        processing_times = [
            2,  # Blow Molding
            2,  # Clean Fill
            6,  # Wrap Heat
            2,  # Robo Arm
        ]
        assembly_condition=1
        source_main = Source(
            'Source_main',
            position=(100, 250),
            processing_time=5,
            carrier_capacity=2,
            actionable_waiting_time=True,
            unlimited_carriers=True,
            carrier_specs={
                'carrier': {"Part": {"Wrap_Heat": {"assembly_condition": assembly_condition}}}
            },
            use_rates=True

        )

        blow_molding = Process(
            'Blow_Molding',
            processing_time=processing_times[0],
            min_processing_time=2,
            actionable_processing_time=True,
            position=(300, 250)
        )

        clean_fill = Process(
            'Clean_Fill',
            processing_time=processing_times[1],
            min_processing_time=2,
            actionable_processing_time=True,
            position=(500, 250)
        )

        wrap_heat = Process(
            'Wrap_Heat',
            processing_time=processing_times[2],
            min_processing_time=6,
            actionable_processing_time=True,
            # processing_std=0.8,
            position=(700, 250)
        )

        robo_arm = Process(
            'Robo_Arm',
            processing_time=processing_times[3],
            min_processing_time=2,
            actionable_processing_time=True,
            position=(900, 250)
        )

        sink = Sink(
            'Sink',
            processing_time=0,
            position=(1100, 250)
        )
        # add multiple machine
        # minimize work in process

        sink.connect_to_input(robo_arm, capacity=20, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        blow_molding.connect_to_input(source_main, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        clean_fill.connect_to_input(blow_molding, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        wrap_heat.connect_to_input(clean_fill, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        robo_arm.connect_to_input(wrap_heat, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)

def visualize_station_states(line, simulation_time=1000):
    """Create a comprehensive dashboard of all station states"""
    
    # Run simulation and collect states
    line.run(simulation_end=simulation_time, visualize=False, capture_screen=False)
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Station State Validation Dashboard', fontsize=16)
    
    stations = [obj for obj in line._objects.values() if hasattr(obj, 'state')]
    
    # 1. Utilization rates
    ax1 = axes[0, 0]
    utilization_data = []
    for station in stations:
        if 'utilization_rate' in station.state.names:
            util_rate = station.state['utilization_rate'].value
            utilization_data.append((station.name, util_rate))
    
    if utilization_data:
        names, rates = zip(*utilization_data)
        ax1.bar(names, rates)
        ax1.set_title('Utilization Rates')
        ax1.set_ylabel('Utilization Rate')
        ax1.tick_params(axis='x', rotation=45)
    else:
        ax1.text(0.5, 0.5, 'No Utilization Data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Utilization Rates')
    
    # 2. Throughput rates
    ax2 = axes[0, 1]
    throughput_data = []
    for station in stations:
        if 'throughput_rate' in station.state.names:
            throughput = station.state['throughput_rate'].value
            throughput_data.append((station.name, throughput))
    
    if throughput_data:
        names, rates = zip(*throughput_data)
        ax2.bar(names, rates)
        ax2.set_title('Throughput Rates')
        ax2.set_ylabel('Parts/Time')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No Throughput Data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Throughput Rates')
    
    # 3. Current window throughput
    ax3 = axes[1, 0]
    window_throughput_data = []
    for station in stations:
        if 'current_window_throughput' in station.state.names:
            window_throughput = station.state['current_window_throughput'].value
            window_throughput_data.append((station.name, window_throughput))
    
    if window_throughput_data:
        names, counts = zip(*window_throughput_data)
        ax3.bar(names, counts)
        ax3.set_title('Current Window Throughput')
        ax3.set_ylabel('Parts Count')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No Window Throughput Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Current Window Throughput')
    
    # 4. Mode distribution - FIX HERE
    ax4 = axes[1, 1]
    mode_counts = {'working': 0, 'waiting': 0, 'failing': 0}
    for station in stations:
        if 'mode' in station.state.names:
            mode = station.state['mode'].value
            if mode in mode_counts:
                mode_counts[mode] += 1
    
    # Filter out zero counts and check if we have any data
    non_zero_modes = {k: v for k, v in mode_counts.items() if v > 0}
    
    if non_zero_modes:
        ax4.pie(non_zero_modes.values(), labels=non_zero_modes.keys(), autopct='%1.1f%%')
        ax4.set_title('Current Mode Distribution')
    else:
        ax4.text(0.5, 0.5, 'No Mode Data Available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Current Mode Distribution')
    
    # 5. Scrap rates (for Assembly stations)
    ax5 = axes[2, 0]
    scrap_data = []
    for station in stations:
        if 'scrap_rate' in station.state.names:
            scrap_rate = station.state['scrap_rate'].value
            scrap_data.append((station.name, scrap_rate))
    
    if scrap_data:
        names, rates = zip(*scrap_data)
        ax5.bar(names, rates)
        ax5.set_title('Scrap Rates')
        ax5.set_ylabel('Scrap/Time')
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5, 0.5, 'No Scrap Data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Scrap Rates')
    
    # 6. State summary table
    ax6 = axes[2, 1]
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create summary table
    summary_data = []
    for station in stations:
        row = [station.name]
        if 'utilization_rate' in station.state.names:
            util_val = station.state['utilization_rate'].value
            if util_val is not None and not (isinstance(util_val, float) and np.isnan(util_val)):
                row.append(f"{util_val:.2f}")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
        
        if 'throughput_rate' in station.state.names:
            throughput_val = station.state['throughput_rate'].value
            if throughput_val is not None and not (isinstance(throughput_val, float) and np.isnan(throughput_val)):
                row.append(f"{throughput_val:.2f}")
            else:
                row.append("N/A")
        else:
            row.append("N/A")
        summary_data.append(row)
    
    if summary_data:
        table = ax6.table(cellText=summary_data, 
                         colLabels=['Station', 'Utilization', 'Throughput'],
                         cellLoc='center', loc='center')
        ax6.set_title('State Summary')
    else:
        ax6.text(0.5, 0.5, 'No Summary Data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('State Summary')
    
    plt.tight_layout()
    plt.show()

def validate_time_series_states(line, simulation_time=1000, sample_interval=10):
    """Track state changes over time to validate moving windows"""
    
    # Create data collectors
    state_history = {
        'time': [],
        'station_states': {}
    }
    
    def collect_states():
        """SimPy process to collect states periodically"""
        while True:
            current_time = line.env.now
            state_history['time'].append(current_time)
            
            for name, obj in line._objects.items():
                if hasattr(obj, 'state'):
                    if name not in state_history['station_states']:
                        state_history['station_states'][name] = {
                            'utilization_rate': [],
                            'throughput_rate': [],
                            'mode': [],
                            'current_window_throughput': []
                        }
                    
                    # Collect key metrics
                    if 'utilization_rate' in obj.state.names:
                        state_history['station_states'][name]['utilization_rate'].append(
                            obj.state['utilization_rate'].value)
                    else:
                        state_history['station_states'][name]['utilization_rate'].append(0)
                    
                    if 'throughput_rate' in obj.state.names:
                        state_history['station_states'][name]['throughput_rate'].append(
                            obj.state['throughput_rate'].value)
                    else:
                        state_history['station_states'][name]['throughput_rate'].append(0)
                    
                    if 'mode' in obj.state.names:
                        mode_val = 1 if obj.state['mode'].value == 'working' else 0
                        state_history['station_states'][name]['mode'].append(mode_val)
                    else:
                        state_history['station_states'][name]['mode'].append(0)
                    
                    if 'current_window_throughput' in obj.state.names:
                        state_history['station_states'][name]['current_window_throughput'].append(
                            obj.state['current_window_throughput'].value)
                    else:
                        state_history['station_states'][name]['current_window_throughput'].append(0)
            
            yield line.env.timeout(sample_interval)
    
    # Start collection process
    line.env.process(collect_states())
    
    # Run simulation
    line.run(simulation_end=simulation_time, visualize=False, capture_screen=False)
    
    # Plot time series
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('State Time Series Validation', fontsize=16)
    
    times = state_history['time']
    
    # Plot utilization over time
    ax1 = axes[0, 0]
    for station_name, data in state_history['station_states'].items():
        if any(data['utilization_rate']):
            ax1.plot(times, data['utilization_rate'], label=station_name, marker='o', markersize=3)
    ax1.set_title('Utilization Rate Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Utilization Rate')
    ax1.legend()
    ax1.grid(True)
    
    # Plot throughput over time
    ax2 = axes[0, 1]
    for station_name, data in state_history['station_states'].items():
        if any(data['throughput_rate']):
            ax2.plot(times, data['throughput_rate'], label=station_name, marker='o', markersize=3)
    ax2.set_title('Throughput Rate Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Throughput Rate')
    ax2.legend()
    ax2.grid(True)
    
    # Plot mode over time (working=1, not working=0)
    ax3 = axes[1, 0]
    for station_name, data in state_history['station_states'].items():
        if any(data['mode']):
            ax3.plot(times, data['mode'], label=station_name, marker='o', markersize=2)
    ax3.set_title('Working Mode Over Time (1=Working, 0=Not Working)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Working Mode')
    ax3.legend()
    ax3.grid(True)
    
    # Plot window throughput
    ax4 = axes[1, 1]
    for station_name, data in state_history['station_states'].items():
        if any(data['current_window_throughput']):
            ax4.plot(times, data['current_window_throughput'], label=station_name, marker='o', markersize=3)
    ax4.set_title('Window Throughput Over Time')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Window Throughput')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return state_history

if __name__ == '__main__':
    line = WaterLine(use_graph_as_states=True)
    # visualize_station_states(line, simulation_time=1000)
    list_of_states =line.run(simulation_end=4000, visualize=False, capture_screen=False)
    # history = validate_time_series_states(line, simulation_time=1000, sample_interval=5)    # import pickle
    # with open('water_line_states.pkl', 'wb') as f:
    #     pickle.dump(list_of_states, f)

    print(line.get_n_parts_produced())
    print(line.get_n_scrap_parts())