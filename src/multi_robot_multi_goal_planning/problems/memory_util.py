import torch

# Global device variable
device = "cuda" if torch.cuda.is_available() else "cpu"

# def check_and_switch_device(memory_threshold: float = 0.001):
#     """
#     Check GPU memory usage and switch to CPU if it exceeds the threshold.
#     :param memory_threshold: Fraction of GPU memory usage (e.g., 0.8 for 80%).
#     """
#     global device

#     if device == "cuda":
#         total_memory = torch.cuda.get_device_properties(0).total_memory
#         allocated_memory = torch.cuda.memory_allocated(0)

#         if allocated_memory / total_memory > memory_threshold:
#             print("Switching to CPU due to memory constraints.")
#             device = "cpu"  # Update the global device variable
#             torch.cuda.empty_cache()  # Free up GPU memory
#             return True
#     return False

def check_gpu_memory_usage(threshold=90):
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated()
        gpu_total = torch.cuda.get_device_properties(0).total_memory
        gpu_usage_percent = (gpu_allocated / gpu_total) * 100

        if gpu_usage_percent >= threshold:
            print(f"GPU memory usage is at {gpu_usage_percent:.2f}%, exceeding {threshold}% threshold.")
            print("\nDetailed GPU Memory Summary:")
            print(torch.cuda.memory_summary())
            return True
        return False


