from mpi4py import MPI
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10000000
dentro = 0

if rank == 0:
    inicio = time.time()

for _ in range(N):
    x = random.random()
    y = random.random()

    if x*x + y*y <= 1:
        dentro += 1

total_dentro = comm.reduce(dentro, op=MPI.SUM, root=0)

if rank == 0:
    total_pontos = N * size
    pi = 4 * total_dentro / total_pontos
    fim = time.time()

    print("PI aproximado:", pi)
    print("Tempo:", (fim - inicio) * 1000, "ms")