import torch
import gc

# Function that serves to offload the SLM after processing to then load the LLM afterwards -- AKA memory swapping
def clearVRAM():
    gc.collect() # Run the garbage collector
    torch.cuda.empty_cache() # Clear CUDA cache
    torch.cuda.reset_peak_memory_stats() # Clear peak memory stats