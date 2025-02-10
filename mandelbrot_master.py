#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from time import time
from PIL import Image
import matplotlib.cm
from math import log

# --- Определение класса множества Мандельброта ---
class MandelbrotSet:
    def __init__(self, max_iterations: int, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius  = escape_radius

    def convergence(self, c: complex, smooth=False) -> float:
        iter_count = self.count_iterations(c, smooth)
        return iter_count / self.max_iterations

    def count_iterations(self, c: complex, smooth=False) -> float:
        # Быстрые проверки для областей, где точка точно входит в множество
        if (c.real * c.real + c.imag * c.imag) < 0.0625:
            return self.max_iterations
        if ((c.real + 1) ** 2 + c.imag * c.imag) < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1j * c.imag
            if abs(ct) < 0.5 * (1 - (ct.real / (abs(ct) if abs(ct) != 0 else 1e-14))):
                return self.max_iterations

        z = 0 + 0j
        for i in range(self.max_iterations):
            z = z * z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return i + 1 - log(log(abs(z))) / log(2)
                return i
        return self.max_iterations

# --- Задание параметров изображения и множества ---
max_iterations = 200
mandelbrot = MandelbrotSet(max_iterations, escape_radius=2.)
width, height = 1024, 1024
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.125, 1.125
scaleX = (x_max - x_min) / width
scaleY = (y_max - y_min) / height

# --- Инициализация MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = 7

if rank == 0:
    # === MASTER PROCESS (rank 0) ===
    print(f"[Master] Starting with {size-1} workers.", flush=True)
    
    t_start = MPI.Wtime()

    full_convergence = np.empty((height, width), dtype=np.double)
    num_tasks = height
    task_index = 0      
    results_received = 0  

    workers_stopped = set()

    for worker in range(1, size):
        if task_index < num_tasks:
            task_index += 1
        else:
            workers_stopped.add(worker)

    # Process results and assign new tasks until all tasks are done.
    while results_received < num_tasks:
        status = MPI.Status()
        worker_source = status.Get_source()
        tag = status.Get_tag()

        if task_index < num_tasks:
            task_index += 1
        else:
            results_received += 1

    for worker in range(1, size):
        if worker not in workers_stopped:
            workers_stopped.add(worker)

    t_compute_end = MPI.Wtime()
    print(f"[Master] calcul: {t_compute_end - t_start:.4f} s", flush=True)

    t_img_start = time()
    t_img_end = time()
    print(f"[Master] constitution de l'image: {t_img_end - t_img_start:.4f} s", flush=True)
    
    #print("[Master] Image saved as 'mandelbrot_dynamic.png'.", flush=True)
    
    comm.Barrier()

else:
    # === WORKER PROCESSES (rank != 0) ===
    while True:
        status = MPI.Status()
        i = 0
        task = i
        tag = status.Get_tag()
        i += 1
        

        line_index = task 
        line_data = np.empty(width, dtype=np.double)
        y = y_min + line_index * scaleY

        # Compute the Mandelbrot set for this row.
        for x in range(width):
            c = complex(x_min + x * scaleX, y)
            line_data[x] = mandelbrot.convergence(c, smooth=True)
