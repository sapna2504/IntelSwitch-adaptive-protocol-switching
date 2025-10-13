import os
import numpy as np
NUM_AGENTS = 3

class Environment:
    def __init__(self, data_tcp, data_quic, agent_id):
        # self.agent_id = agent_id
        # self.data_tcp = data_tcp  # Single agent's TCP data
        # self.data_quic = data_quic  # Single agent's QUIC data
        # self.current_index = 0  # Only for this agent

        self.agent_id = agent_id
        self.current_index = 0
        min_len = min(len(data_tcp), len(data_quic))
        self.data_tcp = data_tcp[:min_len]
        self.data_quic = data_quic[:min_len]
        self.total_rows = min_len
        

    def try_convert(self, item):
        try:
            return float(item)
        except ValueError:
            return item
    
    def get_next_match_cyclic(self, target_str):
        # Sanity check
        # print(self.data_quic)
        if self.data_tcp is None or self.data_quic is None:
            print("TCP or QUIC data not initialized")
            return None, None

        total_rows = min(len(self.data_tcp), len(self.data_quic))
        if total_rows == 0:
            return None, None
        # print(self.data_tcp)
        col2_tcp = self.data_tcp[:, 1]
        col2_quic = self.data_quic[:, 1]
        start_idx = self.current_index

        tcp_row = quic_row = None

        for i in range(total_rows):
            idx = (start_idx + i) % total_rows
            if col2_tcp[idx] == str(target_str):
                tcp_raw = self.data_tcp[idx]
                tcp_row = [self.try_convert(val) for val in tcp_raw[:-2]] + list(tcp_raw[-2:])
            if col2_quic[idx] == str(target_str):
                quic_raw = self.data_quic[idx]
                quic_row = [self.try_convert(val) for val in quic_raw[:-2]] + list(quic_raw[-2:])
            if tcp_row and quic_row:
                self.current_index = (idx + 1) % total_rows
                return tcp_row, quic_row

        print(f"Warning: No match for '{target_str}' in TCP or QUIC for agent {self.agent_id}")
        self.current_index = (self.current_index + 1) % total_rows
        return None, None
        

    # def get_next_match_cyclic(self, target_str, send_data):
    #         print("the send data in env class ",send_data )
    #         # Determine which dataset to search in
    #         if send_data == 0:
    #             # print('Searching in TCP data')
    #             search_data = self.data_tcp
    #         else:
    #             # print('Searching in QUIC data')
    #             search_data = self.data_quic

    #         # Sanity check
    #         if self.data_tcp is None or self.data_quic is None:
    #             print("TCP or QUIC data not initialized")
    #             return None

    #         total_rows = min(len(self.data_tcp), len(self.data_quic))

    #         if total_rows == 0:
    #             return None

    #         col2 = search_data[:, 1]
    #         start_idx = self.current_index

    #         for i in range(total_rows):
    #             idx = (start_idx + i) % total_rows  # cyclic
    #             if col2[idx] == str(target_str):
    #                 self.current_index = (idx + 1) % total_rows  # update shared index
    #                 row = search_data[idx]

    #                 # Convert all except last 2 columns to float
    #                 converted_row = [self.try_convert(val) for val in row[:-2]]
    #                 converted_row += list(row[-2:])
    #                 return converted_row

    #         print(f"Warning: No match for '{target_str}' in {'TCP' if send_data == 0 else 'QUIC'} data for agent {self.agent_id}")
    #         self.current_index = (self.current_index + 1) % total_rows  # still advance the shared index
    #         return None



    
# env = Environment()

# for i in range(3):  # for each agent
#     for j in range(100):
#         row = env.get_last_match_before_true_col13_cyclic(agent_index=i, target_str=str(j))
#         print(f"Agent {i} Row with second column = : {j},{row}")