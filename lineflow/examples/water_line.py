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
            12,  # Wrap Heat
            2,  # Robo Arm
        ]

        source_main = Source(
            'Source_main',
            position=(100, 250),
            processing_time=5,
            carrier_capacity=2,
            actionable_waiting_time=True,
            unlimited_carriers=True,
        )

        blow_molding = Process(
            'Blow_Molding',
            processing_time=processing_times[0],
            min_processing_time=2,
            actionable_processing_time=False,
            position=(300, 250)
        )

        clean_fill = Process(
            'Clean_Fill',
            processing_time=processing_times[1],
            min_processing_time=2,
            actionable_processing_time=False,
            position=(500, 250)
        )

        wrap_heat = Process(
            'Wrap_Heat',
            processing_time=processing_times[2],
            min_processing_time=6,
            actionable_processing_time=False,
            processing_std=0.8,
            position=(700, 250)
        )

        robo_arm = Process(
            'Robo_Arm',
            processing_time=processing_times[3],
            min_processing_time=2,
            actionable_processing_time=False,
            position=(900, 250)
        )

        sink = Sink(
            'Sink',
            processing_time=0,
            position=(1100, 250)
        )
        sink.connect_to_input(robo_arm, capacity=20, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        blow_molding.connect_to_input(source_main, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        clean_fill.connect_to_input(blow_molding, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        wrap_heat.connect_to_input(clean_fill, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)
        robo_arm.connect_to_input(wrap_heat, capacity=50, transition_time=2, min_transition_time=2, controllable_transition_time=True)

    # def build(self):

    #     # Configure a simple line
    #     buffer_1 = Buffer('Buffer1', capacity=50, transition_time=5)
    #     buffer_2 = Buffer('Buffer2', capacity=50, transition_time=5)
    #     buffer_3 = Buffer('Buffer3', capacity=50, transition_time=5)
    #     buffer_4 = Buffer('Buffer4', capacity=50, transition_time=5)
    #     processing_times = [
    #         2,  # Blow Molding
    #         2,  # Clean Fill
    #         12,  # Wrap Heat
    #         2,  # Robo Arm
    #     ]

    #     source_main = Source(
    #         'Source_main',
    #         position=(100, 250),
    #         processing_time=5,
    #         carrier_capacity=2,
    #         actionable_waiting_time=False,
    #         unlimited_carriers=True,
    #         buffer_out=buffer_1
    #     )

    #     blow_molding = Process(
    #         'Blow_Molding',
    #         buffer_in=buffer_1,
    #         buffer_out=buffer_2,
    #         processing_time=processing_times[0],
    #         position=(300, 250)
    #     )

    #     clean_fill = Process(
    #         'Clean_Fill',
    #         buffer_in=buffer_2,
    #         buffer_out=buffer_3,
    #         processing_time=processing_times[1],
    #         position=(500, 250)
    #     )

    #     wrap_heat = Process(
    #         'Wrap_Heat',
    #         buffer_in=buffer_3,
    #         buffer_out=buffer_4,
    #         processing_time=processing_times[2],
    #         position=(700, 250)
    #     )

    #     robo_arm = Process(
    #         'Robo_Arm',
    #         buffer_in=buffer_4,
    #         processing_time=processing_times[3],
    #         position=(900, 250)
    #     )

    #     sink = Sink(
    #         'Sink',
    #         processing_time=0,
    #         position=(1100, 250)
    #     )
    #     sink.connect_to_input(robo_arm, capacity=20, transition_time=2)

if __name__ == '__main__':
    line = WaterLine(use_graph_as_states=True)
    line.run(simulation_end=4000, visualize=True, capture_screen=False)

    print(line.get_n_parts_produced())