# File: rebuild_nccl_utils.py
import os
import time
import json
import pickle
import torch
import torch.distributed as dist
import signal

class NCCLRebuildUtils:

    def __init__(self):
        self.needRebuild = False
        self.groupAbuilt=False
        self.groupBbuilt=False
        self.groupA=None
        self.groupB=None
        signal.signal(signal.SIGUSR1, self.rebuilt_handler)
        
    def rebuilt_handler(self, signum):
        self.needRebuild = True
        print(f"rebuild signal received{signum}")
        pass
    
    def rebuild(self,ranks):
        del self.groupB
        self.needRebuild = False
        self.groupB = dist.new_group(ranks=ranks)
        
    def buildGroupA(self):
        if not self.groupAbuilt:
            self.groupAbuilt = True
            ranks = torch.distributed.get_world_size()
            self.groupA = dist.new_group(ranks=range(ranks))
    
    def buildGroupB(self):
        if not self.groupBbuilt:
            self.groupBbuilt = True
            ranks = torch.distributed.get_world_size()
            self.groupB = dist.new_group(ranks=range(ranks))
            
        
    @staticmethod
    def init_process_group(temp_dir="/home/ubuntu/Codespace/serverless-moe/mixtral/REPLICATE/saved_objects/rank_0/initialize_distributed.pickle"):
        if not dist.is_initialized():
            with open(temp_dir, 'rb') as f:
                params_data = pickle.load(f)
            get_embedding_ranks = params_data['get_embedding_ranks']
            get_position_embedding_ranks = params_data['get_position_embedding_ranks']
            _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

    @staticmethod
    def save_nccl_state(file_path):
        if dist.is_initialized():
            state = {
                'world_size': dist.get_world_size(),
                'rank': dist.get_rank(),
                'backend': dist.get_backend(),
            }
            with open(file_path, 'w') as f:
                json.dump(state, f)
            print(f"[Rank {state['rank']}] Saved NCCL state to {file_path}.")

    @staticmethod
    def load_nccl_state(file_path):
        with open(file_path, 'r') as f:
            state = json.load(f)
        print(f"[Rank {state['rank']}] Loaded NCCL state from {file_path}.")
        return state

    @staticmethod
    def cleanup():
        if dist.is_initialized():
            rank = dist.get_rank()
            dist.destroy_process_group()
            print(f"[Rank {rank}] Destroyed group.")


ncclrebuildutils = NCCLRebuildUtils()
