from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = 1

# Dimension du problème (doit être divisible par nbp)
N = 100
Nloc = N // nbp

comm.Barrier()
t_total_start = MPI.Wtime()
# Construction du bloc de lignes pour chaque tâche
rows = np.arange(rank * Nloc, (rank + 1) * Nloc)
A_local = np.array([[(i + j) % N + 1. for i in range(N)] for j in rows])
u = np.array([i + 1. for i in range(N)])  # vecteur u complet sur chaque tâche

v_local = A_local.dot(u)


# Rassemblement des résultats de chaque tâche pour obtenir le vecteur complet v
v = np.empty(N, dtype=np.float64)
comm.Allgather(v_local, v,)

comm.Barrier()
t_total_end = MPI.Wtime()
t_total = t_total_end - t_total_start

# Affichage des résultats et des temps (affiché par la tâche 0)
if rank == 0:
    print("Produit matrice-vecteur par ligne:")
    #print("v =", v)
    print(f"Temps de calcul: {t_total:.6f} seconds")