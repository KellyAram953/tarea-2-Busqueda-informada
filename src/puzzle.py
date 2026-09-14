import heapq
import time
import numpy as np
from collections import deque

# =========================== 8-PUZZLE ================================
# Convierte la matriz 3×3 a una tupla plana de 9 elementos.
def _puzzle_vec(m):
    return tuple(int(x) for x in np.asarray(m).flatten())

def _puzzle_sucesores(estado):
    i = estado.index(0)                   # posición del hueco 
    f, c = divmod(i, 3)                  # fila, columna del hueco
    # Prueba las 4 direcciones
    for _, df, dc in (('ARRIBA', -1, 0), ('ABAJO', 1, 0), ('IZQUIERDA', 0, -1), ('DERECHA', 0, 1)):
        nf, nc = f + df, c + dc                     # nueva posición
        if 0 <= nf < 3 and 0 <= nc < 3:             
            j = nf * 3 + nc                         # índice en la tupla plana
            nuevo = list(estado)                    # copia mutable
            nuevo[i], nuevo[j] = nuevo[j], nuevo[i] # intercambia hueco y ficha
            yield tuple(nuevo)                      # devuelve la nueva tupla

# ================Heurística para el 8-Puzzle.==========================0

def _h_puzzle(estado, objetivo):
    pos = {v: k for k, v in enumerate(objetivo)}    
    d = 0
    for i, v in enumerate(estado):
        if v == 0:             # ignora el hueco
            continue
        j = pos[v]          # posición meta de la ficha
        # |fila_actual - fila_meta| + |col_actual - col_meta|
        d += abs(i // 3 - j // 3) + abs(i % 3 - j % 3)
    return d

# Dibujar la animación paso a paso con `explorar_tableros()`.
def _camino_a_tableros(camino):
    return [np.array(e).reshape(3, 3) for e in camino]

# =========Busqueda por anchura para el 8-Puzzle.============
def _puzzle_bfs(datos, config):
    ini  = _puzzle_vec(datos)                  # estado inicial
    meta = tuple(config['objetivo'])            # estado objetivo 
    lim_exp = config.get('limite_expandidos', 200000)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()
    fr = deque([(ini, [ini])])                     
    visit = {ini}
    exp = gen = 1
    fmax = 1

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)
        est, camino = fr.popleft()                  
        exp += 1
        if est == meta:
            return _res(costo=len(camino)-1,        # costo = movimientos
                        expandidos=exp, generados=gen, frontera_max=fmax,
                        profundidad=len(camino)-1,
                        visual={'tableros': _camino_a_tableros(camino)})
        for nxt in _puzzle_sucesores(est):
            if nxt not in visit:
                visit.add(nxt)
                gen += 1
                fr.append((nxt, camino + [nxt]))
        fmax = max(fmax, len(fr))
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)

# ==============Busqueda por profundidad ara el 8-Puzzle=============
# Sin límite puede caer en caminos infinitos para evitarlo se corta cuando d >= prof_max.
def _puzzle_dfs(datos, config):
    ini  = _puzzle_vec(datos)
    meta = tuple(config['objetivo'])
    lim_exp  = config.get('limite_expandidos', 200000)
    lim_s    = config.get('limite_s', 10)
    prof_max = config.get('limite_profundidad', 25)     #corte de profundidad
    t0 = time.perf_counter()
  # Pila: (estado, camino, profundidad, visitados-en-rama)
    fr = [(ini, [ini], 0, {ini})]
    exp = gen = 0
    fmax = 1
    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen,
                        frontera_max=fmax, notas='límite alcanzado')
        est, camino, d, visto = fr.pop()          
        exp += 1
        if est == meta:
            return _res(costo=len(camino)-1, expandidos=exp, generados=gen,
                        frontera_max=fmax, profundidad=len(camino)-1,
                        visual={'tableros': _camino_a_tableros(camino)})

        if d >= prof_max:           # si llegó al tope, no expande
            continue
        for nxt in _puzzle_sucesores(est):
            if nxt not in visto:
                gen += 1
                fr.append((nxt, camino + [nxt], d + 1, visto | {nxt}))
        fmax = max(fmax, len(fr))
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)


# ============Busqueda de costo uniforme para el 8-Puzzle.=========0
def _puzzle_ucs(datos, config):
    ini  = _puzzle_vec(datos)
    meta = tuple(config['objetivo'])
    lim_exp = config.get('limite_expandidos', 200000)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()

    fr = [(0, 0, ini, [ini])]      # (g, contador, estado, camino)
    mejor = {ini: 0}
    cont = 0
    exp = gen = 0
    fmax = 1

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)
        g, _, est, camino = heapq.heappop(fr)
        exp += 1
        if est == meta:
            return _res(costo=g, expandidos=exp, generados=gen,
                        frontera_max=fmax, profundidad=len(camino)-1,
                        visual={'tableros': _camino_a_tableros(camino)})

        for nxt in _puzzle_sucesores(est):
            ng = g + 1            # cada movimiento cuesta 1
            if nxt not in mejor or ng < mejor[nxt]:
                mejor[nxt] = ng
                cont += 1
                gen += 1
                heapq.heappush(fr, (ng, cont, nxt, camino + [nxt]))
        fmax = max(fmax, len(fr))
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)
