# All-Pairs Shortest Path Optimization
## hw3-1: CPU Threading with pthread Library
- Allocated contiguous chunks of data to each thread to improve cache performance
- Used `pthread library` to evenly distribute computation across CPU cores
- Inserted `pthread_barrier_wait` to synchronize all threads at the end of each vertex’s computation step

## hw3-2: CUDA Optimization — Single GPU Version
- Use CUDA to optimize the APSP algorithm on a single GPU using the **Blocked Floyd-Warshall algorithm**
  - Implemented the three computation phases:
    - phase1: Independent computation within the pivot block
    - phase2: Computation for the pivot row and pivot column blocks
    - phase3: Computation for the remaining non-pivot blocks
- Applied the following CUDA optimization strategies:
    1. **Block Factor Calculation**: Determined based on shared memory size to ensure that all data for a block can fit within shared memory, avoiding spills to slower global memory.
    2. **Thread Workload Distribution**: When block factor = 64, each 64×64 block contains 4096 elements. Since each CUDA block supports max 1024 threads, each thread processes 4 elements
    3. **Memory Padding**: Applied padding to prevent shared memory bank conflicts
## hw3-3: CUDA Optimization — Multi-GPU Version
- Extend the CUDA version to run on multiple GPUs using OpenMP and thread management
- Built upon the HW3-2 implementation with multi-threading and OpenMP
- Assigned one thread per GPU, allowing two threads to manage two GPUs concurrently
- Due to **inter-phase dependencies**:
  - Both GPUs execute Phase 1 and Phase 2 to maintain data consistency
  - Phase 3, which has the highest computational load, is evenly split between the two GPUs
- Minimized inter-GPU communication to reduce overhead, only synchronizing where necessary
