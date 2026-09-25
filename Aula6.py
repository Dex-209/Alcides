import argparse
import socket
import sys
import time

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
ROOT = 0
PCT_ATENCAO = 1.0   
SEED = 42       

def ler_argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--linhas", type=int, default=2000)
    p.add_argument("--colunas", type=int, default=2000)
    p.add_argument("--limiar", type=int, default=200)
    p.add_argument("--limiar-alto", type=int, default=230)
    p.add_argument("--pct-critico", type=float, default=5.0)
    p.add_argument("--sleep", type=float, default=1.0)
    return p.parse_args()


def gerar_radiografia(linhas, colunas, seed):

    rng = np.random.default_rng(seed)
    y, x = np.ogrid[0:linhas, 0:colunas]
    yn = y / linhas         
    xn = x / colunas

    img = np.full((linhas, colunas), 25.0, dtype=np.float32)    
    torax = ((xn - 0.5) / 0.42) ** 2 + ((yn - 0.52) / 0.45) ** 2 <= 1
    img = np.where(torax, 135.0, img)
    pulmao_esq = ((xn - 0.30) / 0.16) ** 2 + ((yn - 0.48) / 0.33) ** 2 <= 1
    pulmao_dir = ((xn - 0.70) / 0.16) ** 2 + ((yn - 0.48) / 0.33) ** 2 <= 1
    img = np.where(pulmao_esq | pulmao_dir, 65.0, img)
    coluna = (np.abs(xn - 0.5) < 0.035) & torax
    img = np.where(coluna, 175.0, img)
    img += rng.normal(0, 8, size=(linhas, colunas)).astype(np.float32)
    n_esq = int(rng.integers(3, 6))
    n_dir = int(rng.integers(6, 10))
    for cx_centro, n in ((0.30, n_esq), (0.70, n_dir)):
        for _ in range(n):
            cx = cx_centro + rng.uniform(-0.10, 0.10)
            cy = 0.48 + rng.uniform(-0.25, 0.25)
            raio = rng.uniform(0.015, 0.045)
            pico = rng.uniform(150, 200)  
            y0, y1 = max(0, int((cy - 3 * raio) * linhas)), min(linhas, int((cy + 3 * raio) * linhas) + 1)
            x0, x1 = max(0, int((cx - 3 * raio) * colunas)), min(colunas, int((cx + 3 * raio) * colunas) + 1)
            d2 = ((xn[:, x0:x1] - cx) ** 2 + (yn[y0:y1, :] - cy) ** 2) / (raio ** 2)
            img[y0:y1, x0:x1] += (pico * np.exp(-d2)).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)

def dividir_linhas(linhas, nprocs):
    base, resto = divmod(linhas, nprocs)
    faixas, inicio = [], 0
    for r in range(nprocs):
        qtd = base + (1 if r < resto else 0)
        faixas.append((inicio, qtd))
        inicio += qtd
    return faixas


def classificar_faixa(pct, pct_atencao, pct_critico):
    if pct > pct_critico:
        return "CRITICA"
    if pct >= pct_atencao:
        return "ATENCAO"
    return "NORMAL"


def classificar_exame(pct_global, pct_atencao, pct_critico, faixas_criticas):
    if pct_global > pct_critico or faixas_criticas >= max(1, size // 2):
        return "ALTA CONCENTRACAO DE AREAS SUSPEITAS"
    if pct_global >= pct_atencao or faixas_criticas > 0:
        return "ATENCAO CLINICA"
    return "SEM INDICIOS RELEVANTES"


def main():
    args = ler_argumentos()
    t_inicio = MPI.Wtime()

    imagem = None
    params = None
    if rank == ROOT:
        t0 = MPI.Wtime()
        imagem = gerar_radiografia(args.linhas, args.colunas, SEED)
        t_geracao = MPI.Wtime() - t0
        params = {
            "linhas": args.linhas,
            "colunas": args.colunas,
            "limiar": args.limiar,
            "limiar_alto": args.limiar_alto,
            "pct_critico": args.pct_critico,
            "pct_atencao": PCT_ATENCAO,
            "sleep": args.sleep,
        }

    params = comm.bcast(params, root=ROOT)
    comm.Barrier()
    t_analise_inicio = MPI.Wtime()

    linhas, colunas = params["linhas"], params["colunas"]
    faixas = dividir_linhas(linhas, size)
    linha_ini, minhas_linhas = faixas[rank]

    contagens = [qtd * colunas for (_, qtd) in faixas]          
    deslocamentos = [ini * colunas for (ini, _) in faixas]       
    minha_faixa = np.empty((minhas_linhas, colunas), dtype=np.uint8)

    t0 = MPI.Wtime()
    if rank == ROOT:
        envio = [imagem, contagens, deslocamentos, MPI.UNSIGNED_CHAR]
    else:
        envio = None
    comm.Scatterv(envio, minha_faixa, root=ROOT)
    t_scatter = MPI.Wtime() - t0

    t0 = MPI.Wtime()
    meio = colunas // 2
    suspeito = minha_faixa > params["limiar"]
    alto = minha_faixa > params["limiar_alto"]

    pixels = int(minha_faixa.size)
    soma = int(minha_faixa.sum(dtype=np.int64))
    maximo = int(minha_faixa.max()) if pixels else 0
    media = soma / pixels if pixels else 0.0
    n_susp = int(suspeito.sum())
    n_alto = int(alto.sum())
    n_esq = int(suspeito[:, :meio].sum())      
    n_dir = int(suspeito[:, meio:].sum())     
    pct_local = 100.0 * n_susp / pixels if pixels else 0.0
    classe = classificar_faixa(pct_local, params["pct_atencao"], params["pct_critico"])
    t_calculo = MPI.Wtime() - t0

    lento = params["sleep"] > 0 and rank % 2 == 1    
    if lento:
        time.sleep(params["sleep"])
    t_local_total = MPI.Wtime() - t_analise_inicio

    comm.Barrier()
    t_barreira = MPI.Wtime() - t_analise_inicio

    g_pixels = comm.reduce(pixels, op=MPI.SUM, root=ROOT)
    g_soma = comm.reduce(soma, op=MPI.SUM, root=ROOT)
    g_susp = comm.reduce(n_susp, op=MPI.SUM, root=ROOT)
    g_alto = comm.reduce(n_alto, op=MPI.SUM, root=ROOT)
    g_esq = comm.reduce(n_esq, op=MPI.SUM, root=ROOT)
    g_dir = comm.reduce(n_dir, op=MPI.SUM, root=ROOT)
    g_max = comm.reduce(maximo, op=MPI.MAX, root=ROOT)

    relatorio = {
        "rank": rank,
        "host": socket.gethostname(),
        "linhas": (linha_ini, linha_ini + minhas_linhas - 1),
        "qtd_linhas": minhas_linhas,
        "media": media,
        "maximo": maximo,
        "suspeitos": n_susp,
        "altos": n_alto,
        "esq": n_esq,
        "dir": n_dir,
        "pct": pct_local,
        "classe": classe,
        "lento": lento,
        "t_scatter": t_scatter,
        "t_calculo": t_calculo,
        "t_local": t_local_total,
    }
    relatorios = comm.gather(relatorio, root=ROOT)

    t_total = MPI.Wtime() - t_inicio

    if rank == ROOT:
        imprimir_relatorio(params, relatorios, t_geracao, t_barreira, t_total,
                           g_pixels, g_soma, g_susp, g_alto, g_esq, g_dir, g_max)


def imprimir_relatorio(p, rels, t_geracao, t_analise, t_total,
                       g_pixels, g_soma, g_susp, g_alto, g_esq, g_dir, g_max):
    media_global = g_soma / g_pixels
    pct_susp = 100.0 * g_susp / g_pixels
    pct_alto = 100.0 * g_alto / g_pixels
    criticas = sum(1 for r in rels if r["classe"] == "CRITICA")
    exame = classificar_exame(pct_susp, p["pct_atencao"], p["pct_critico"], criticas)

    if g_esq > g_dir:
        lado = "esquerdo"
    elif g_dir > g_esq:
        lado = "direito"
    else:
        lado = "empate"

    print("RELATORIO FINAL")
    print()
    print("Informacoes gerais")
    print(f"- Tamanho da imagem: {p['linhas']} x {p['colunas']}")
    print(f"- Numero de processos: {size}")
    print(f"- Limiar de suspeita leve: {p['limiar']}")
    print(f"- Limiar de suspeita alta: {p['limiar_alto']}")
    print(f"- Percentual critico: {p['pct_critico']}%")
    print(f"- Sleep nos processos impares: {p['sleep']} s")
    print(f"- Tempo de geracao da imagem: {t_geracao:.4f} s")
    print(f"- Tempo da analise paralela: {t_analise:.4f} s")
    print(f"- Tempo total: {t_total:.4f} s")
    print()
    print("Estatisticas globais")
    print(f"- Pixels analisados: {g_pixels}")
    print(f"- Media global: {media_global:.2f}")
    print(f"- Intensidade maxima: {g_max}")
    print(f"- Pixels suspeitos: {g_susp} ({pct_susp:.2f}%)")
    print(f"- Pixels altamente suspeitos: {g_alto} ({pct_alto:.2f}%)")
    print()
    print("Comparacao entre os pulmoes")
    print(f"- Suspeitos no lado esquerdo: {g_esq}")
    print(f"- Suspeitos no lado direito: {g_dir}")
    print(f"- Lado com mais areas suspeitas: {lado}")
    print()
    print("Resultados por processo")
    for r in rels:
        ini, fim = r["linhas"]
        print(f"- Processo {r['rank']}: linhas {ini}-{fim}, suspeitos {r['suspeitos']}, "
              f"maximo {r['maximo']}, {r['pct']:.2f}%, {r['classe']}")
    print()
    print(f"Classificacao final do exame: {exame}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()