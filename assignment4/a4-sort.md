````markdown
# Hybrid MPI + FastFlow MergeSort

**Assignment 4**
SPM course a.a. 24/25

May 14 2025

Design and implement a scalable MergeSort for an N-element array of fixed-size records. Each record has the following layout:

```cpp
struct Record {
    unsigned long key; // sorting value
    char rpayload[RPAYLOAD];
};
```

The MPI part handles inter-node distribution and merging, whereas the FastFlow part provides intra-node MergeSort parallelization of local partitions.

## Tasks

1.  **Single-node version (shared-memory)**
    Provide a parallel implementation for a single node using FastFlow building blocks (i.e., farm, pipeline, and all-to-all) of the MergeSort algorithm.

2.  **Multi-node hybrid version**
    Provide a hybrid parallel implementation using MPI and FastFlow. The intra-node parallel MergeSort should reuse what was developed in Task 1. The inter-node MPI communications of the merging phase should try to maximize the opportunity of computation-to-communication overlap.

3.  **Performance study and discussion**
    Analyze the performance by varying the problem size N, the record payload, and the number of FastFlow threads. Report speedup and efficiency varying the number of threads on a single node, and strong and weak scalability curves on the spmcluster up to 8 nodes. Summarize bottleneck phases, overlap effectiveness, challenges encountered, and optimizations you adopted.

**Command line options to consider for both parallel versions:**

- `-s N`: array size (e.g., -s 10M, -s 100M)
- `-r R`: record payload (in bytes, e.g., -r 8, -r 64, -r 256)
- `-t T`: number of FastFlow threads (e.g., -t 16, -t 32)

All parallel versions developed should aim to minimize the overhead.

## Deliverables

Provide all source files, scripts to compile and execute your code on the cluster nodes, and a PDF report (max 5-6 pages) including a brief description of your implementations and the performance analysis conducted. Mention the challenges encountered and the solutions adopted. Submit by email your source code and PDF report in a single zip file named 'sort_parallel\_\<YourName\>.zip' by June 1 EOB. Please use the email subject "SPM Assignment 4".

```

```
````
