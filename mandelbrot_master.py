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

# Теги для управления сообщениями
TAG_WORK = 1
TAG_DONE = 3
TAG_STOP = 2

if rank == 0:
    # === MASTER PROCESS (rank 0) ===
    print(f"[Master] Starting with {size-1} workers.", flush=True)
    
    # Start overall computation timing using MPI.Wtime()
    t_start = MPI.Wtime()

    # Array to store computed values for each row
    full_convergence = np.empty((height, width), dtype=np.double)
    num_tasks = height  # one task per row
    task_index = 0      # index for the next row to compute
    results_received = 0  # count of computed rows received from workers

    # Set to track which workers have been sent a stop signal
    workers_stopped = set()

    # Initially send one task to each worker (if available)
    for worker in range(1, size):
        if task_index < num_tasks:
            print(f"[Master] Sending task {task_index} to worker {worker}.", flush=True)
            comm.send(task_index, dest=worker, tag=TAG_WORK)
            task_index += 1
        else:
            print(f"[Master] No initial task for worker {worker}. Sending stop signal.", flush=True)
            comm.send(None, dest=worker, tag=TAG_STOP)
            workers_stopped.add(worker)

    # Process results and assign new tasks until all tasks are done.
    while results_received < num_tasks:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker_source = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_DONE:
            line_index = result['line_index']
            full_convergence[line_index, :] = result['line_data']
            results_received += 1
            print(f"[Master] Received result for row {line_index} (total results: {results_received}/{num_tasks}) from worker {worker_source}.", flush=True)

            # If there are more tasks, assign the next one.
            if task_index < num_tasks:
                print(f"[Master] Sending task {task_index} to worker {worker_source}.", flush=True)
                comm.send(task_index, dest=worker_source, tag=TAG_WORK)
                task_index += 1
            else:
                # No more tasks remain. If we haven't already told this worker to stop, do so now.
                if worker_source not in workers_stopped:
                    print(f"[Master] No more tasks. Sending stop signal to worker {worker_source}.", flush=True)
                    comm.send(None, dest=worker_source, tag=TAG_STOP)
                    workers_stopped.add(worker_source)
        else:
            print(f"[Master] Received message with unexpected tag {tag} from worker {worker_source}.", flush=True)
    
    # Ensure that every worker has been sent a stop signal.
    for worker in range(1, size):
        if worker not in workers_stopped:
            print(f"[Master] Sending final stop signal to worker {worker}.", flush=True)
            comm.send(None, dest=worker, tag=TAG_STOP)
            workers_stopped.add(worker)

    t_compute_end = MPI.Wtime()
    print(f"[Master] Total computation time: {t_compute_end - t_start:.4f} s", flush=True)

    # Measure image formation time using time() from the Python standard library.
    t_img_start = time()
    # Transpose the array so that matplotlib sees (height, width)
    image = Image.fromarray(np.uint8(matplotlib.cm.cm.plasma(full_convergence.T) * 255))
    t_img_end = time()
    print(f"[Master] Image formation time: {t_img_end - t_img_start:.4f} s", flush=True)
    
    image.save("mandelbrot_dynamic.png")
    print("[Master] Image saved as 'mandelbrot_dynamic.png'.", flush=True)
    
    # Optional barrier to ensure all processes reach here before finishing
    comm.Barrier()

else:
    # === WORKER PROCESSES (rank != 0) ===
    print(f"[Worker {rank}] Started.", flush=True)
    while True:
        status = MPI.Status()
        # Receive a task from the master.
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == TAG_STOP or task is None:
            print(f"[Worker {rank}] Received stop signal. Exiting.", flush=True)
            break


        line_index = task  # row index to compute
        print(f"[Worker {rank}] Received task for row {line_index}.", flush=True)
        line_data = np.empty(width, dtype=np.double)
        y = y_min + line_index * scaleY

        # Compute the Mandelbrot set for this row.
        for x in range(width):
            c = complex(x_min + x * scaleX, y)
            line_data[x] = mandelbrot.convergence(c, smooth=True)

        # Send the computed row back to the master.
        comm.send({'line_index': line_index, 'line_data': line_data},
                  dest=0, tag=TAG_DONE)
        print(f"[Worker {rank}] Completed task for row {line_index} and sent result.", flush=True)
    
    # Optional barrier before worker exit
    comm.Barrier()