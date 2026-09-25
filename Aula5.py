from mpi4py import MPI
import random

# Inicialização do MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -----------------------------
# Função para gerar logs
# -----------------------------
def gerar_logs(qtd):
    ips = [f"192.168.1.{i}" for i in range(1, 255)]

    endpoints = [
        "/",
        "/login",
        "/products",
        "/cart",
        "/checkout",
        "/api/users",
        "/api/orders"
    ]

    metodos = ["GET", "POST"]

    status = ["200", "200", "200", "404", "500"]

    logs = []

    for _ in range(qtd):
        ip = random.choice(ips)
        metodo = random.choice(metodos)
        endpoint = random.choice(endpoints)
        codigo = random.choice(status)

        logs.append(f"{ip} {metodo} {endpoint} {codigo}")

    return logs


# -----------------------------
# Processo 0 gera o dataset
# -----------------------------
logs_divididos = None
TOTAL_LOGS = 1000000

if rank == 0:
    print("\nGerando dataset de logs...\n")

    logs = gerar_logs(TOTAL_LOGS)

    # Garante divisão igual entre os processos
    linhas_por_processo = TOTAL_LOGS // size
    logs = logs[:linhas_por_processo * size]

    # Divide em partes iguais
    logs_divididos = [
        logs[i * linhas_por_processo:(i + 1) * linhas_por_processo]
        for i in range(size)
    ]

# -----------------------------
# Distribuição usando Scatter
# -----------------------------
logs_locais = comm.scatter(logs_divididos, root=0)

# -----------------------------
# Processamento local
# -----------------------------
erros = 0

for linha in logs_locais:
    campos = linha.split()
    status = campos[3]

    if status in ("404", "500"):
        erros += 1

# -----------------------------
# Resultado local
# -----------------------------
print(
    f"Processo {rank} analisou {len(logs_locais)} linhas "
    f"e encontrou {erros} erros."
)