from lineflow.simulation import (
    Buffer,
    Source,
    Sink,
    Line,
    Process,
    Magazine,
)


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

if __name__ == '__main__':
    line = WaterLine(use_graph_as_states=True)
    line.run(simulation_end=4000, visualize=False, capture_screen=False)

    print(line.get_n_parts_produced())
    print(line.get_n_scrap_parts())