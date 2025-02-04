# Produit matrice-vecteur v = A.u
import numpy as np
import time
# Dimension du problème (peut-être changé)
dim = 500

start = time.time()

# Initialisation de la matrice
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
#print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
#print(f"u = {u}")

# Produit matrice-vecteur

#v = A.dot(u)
end = time.time()

exec = end - start

#print(f"v = {v}")
print(f"Temps d'execution: {exec:.6f} seconds")