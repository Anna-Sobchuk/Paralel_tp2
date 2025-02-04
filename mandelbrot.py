#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
from time import time
from PIL import Image
import matplotlib.cm
from math import log

num_threads = 1

class MandelbrotSet:
    def __init__(self, max_iterations: int, escape_radius: float = 2.0):
        self.max_iterations = max_iterations
        self.escape_radius  = escape_radius

    def convergence(self, c: complex, smooth=False) -> float:
        iter_count = self.count_iterations(c, smooth)
        return iter_count / self.max_iterations

    def count_iterations(self, c: complex, smooth=False) -> float:
        if (c.real*c.real + c.imag*c.imag) < 0.0625:
            return self.max_iterations
        if ((c.real+1)**2 + c.imag*c.imag) < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1j * c.imag
            if abs(ct) < 0.5 * (1 - (ct.real / (abs(ct) if abs(ct) != 0 else 1e-14))):
                return self.max_iterations

        z = 0 + 0j
        for i in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return i + 1 - log(log(abs(z)))/log(2)
                return i
        return self.max_iterations


max_iterations = 200
mandelbrot = MandelbrotSet(max_iterations, escape_radius=2.)
width, height = 1024, 1024
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.125, 1.125
scaleX = (x_max - x_min) / width
scaleY = (y_max - y_min) / height


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = num_threads 


rows_per_proc = height // size
extra = height % size

if rank < extra:
    start = rank * (rows_per_proc + 1)
    end = start + rows_per_proc + 1
else:
    start = rank * rows_per_proc + extra
    end = start + rows_per_proc


local_convergence = np.empty((width, end - start), dtype=np.double)
t_deb = MPI.Wtime()
for j_local, j in enumerate(range(start, end)):
    y = y_min + j * scaleY
    for x in range(width):
        c = complex(x_min + x * scaleX, y)
        local_convergence[x, j_local] = mandelbrot.convergence(c, smooth=True)
t_fin = MPI.Wtime()
local_time = t_fin - t_deb


if rank == 0:
    full_convergence = np.empty((width, height), dtype=np.double)
else:
    full_convergence = None


sendcounts = np.array(comm.gather(local_convergence.shape[1], root=0))
if rank == 0:
    displs = np.insert(np.cumsum(sendcounts[:-1]), 0, 0)
else:
    displs = None


comm.Gatherv(sendbuf=local_convergence.flatten(), 
             recvbuf=(full_convergence.flatten() if rank==0 else None, 
                      (sendcounts*width, displs*width)), root=0)


if rank == 0:
    full_convergence = full_convergence.reshape((width, height))
    total_time = MPI.Wtime() - t_deb  
    print(f"[Block] calcul par {num_threads} threads = {max(comm.gather(local_time, root=0)):.4f} с")
    
    t_img_deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(full_convergence.T)*255))
    t_img_fin = time()
    print(f"[Block] construction de l'image: {t_img_fin-t_img_deb:.4f} с")
    image.save("mandelbrot_bloc.png")