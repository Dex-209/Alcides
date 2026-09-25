import random
import time
N = 10000000
dentro = 0
inicio = time.time()
for _ in range(N):
    x = random.random()
    y = random.random()
    if x*x + y*y <= 1:
        dentro += 1
pi = 4 * dentro / N
fim = time.time()
print("PI aproximado:", pi)
print("Tempo:", (fim-inicio)*1000, "ms")