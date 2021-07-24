# Calculating Trigonometric Ratio with GPU

Calculating trigonometric ratio with GPU (using CUDA APIs)

It took 17.6 microseconds to solve sin, cos, tan value from 0 to 999999 (with AMD Ryzen 3800X & NVIDIA RTX 2060 SUPER)

At first, I planned to compare the run time between cpu and gpu.
I measured GPU run time by <code>nvprof</code>, but I could'nt measure the CPU run time precisely.

![NVPROF Result]()