from mpi4py import MPI
import random
import time


N = 1000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]
    inicio = time.time()
else:
    A = None
    B = None

A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

linhas = N // size
inicio_linha = rank * linhas

if rank == size - 1:
    fim_linha = N
else:
    fim_linha = inicio_linha + linhas

C_local = []

for i in range(inicio_linha, fim_linha):
    linha = []
    for j in range(N):
        soma = 0
        for k in range(N):
            soma += A[i][k] * B[k][j]
        linha.append(soma)
    C_local.append(linha)

partes = comm.gather(C_local, root=0)

if rank == 0:
    C = []

    for parte in partes:
        C.extend(parte)

    fim = time.time()

    print(f"Tempo MPI: {(fim - inicio) * 1000:.2f} ms")