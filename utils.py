import torch
import gc
import logging

# Function that monitors current VRAM usage without clearing cache
def checkVRAM():
    vramUsed = torch.cuda.memory_allocated() / (1024 ** 3)
    vramTotal = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logging.info("--- VRAM Monitor")
    logging.info(" -- Used VRAM: {vramUsed:.2f} / {vramTotal:.2f} GB")
    print("----")
    return vramUsed, vramTotal 

# Function that serves to offload the SLM after processing to then load the LLM afterwards -- AKA memory swapping
def clearVRAM():
    vramAvailable = torch.cuda.memory_allocated() / (1024 ** 3) # Amount of memory currently being used before clearance (in GB)

    gc.collect() # Run the garbage collector
    torch.cuda.empty_cache() # Clear CUDA cache
    torch.cuda.reset_peak_memory_stats()

    vramFinal = torch.cuda.memory_allocated() / (1024 ** 3)
    vramTotal = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    logging.info(f"--- Emptying VRAM")
    logging.info(f" -- Memory Cleared: {vramAvailable - vramFinal:.2f} GB")
    logging.info(f" -- Current Occupied Memory: {vramFinal:.2f} GB")
    logging.info(f" -- Free Memory: {vramTotal - vramFinal:.2f} / {vramTotal:.2f} GB")
    print("----")
