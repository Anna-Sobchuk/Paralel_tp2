from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = 1

# Dimension du problème (doit être divisible par nbp)
N = 1000
Nloc = N // nbp

comm.Barrier()
t_total_start = MPI.Wtime()

# Construction du bloc de colonnes pour chaque tâche
cols = np.arange(rank * Nloc, (rank + 1) * Nloc)
# On ne construit que le bloc de colonnes dont la tâche a besoin
A_local = np.array([[(i + j) % N + 1. for i in cols] for j in range(N)])
u = np.array([i + 1. for i in range(N)])  # vecteur u complet sur chaque tâche


v_local = A_local.dot(u[cols])

# Réduction pour assembler le vecteur complet v sur toutes les tâches
v = np.empty(N, dtype=np.float64)
comm.Allreduce(v_local, v, op=MPI.SUM)

comm.Barrier()
t_total_end = MPI.Wtime()
t_total = t_total_end - t_total_start

# Affichage des résultats et des temps (affiché par la tâche 0)
if rank == 0:
    print("Produit matrice-vecteur par colonne:")
    #print("v =", v)
    print(f"Temps de calcul: {t_total:.6f} seconds")