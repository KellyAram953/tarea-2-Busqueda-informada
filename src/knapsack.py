import heapq
import time
from collections import deque

# =========================== KNAPSACK ================================
def _h_knapsack(datos, idx, cap_rest):
    w, p = datos['Weights'], datos['Prices']
    # Ordena los ítems 
    items = sorted([(p[k]/w[k], w[k], p[k]) for k in range(idx, len(w))],
                   reverse=True)
    val, libre = 0.0, cap_rest
    for ratio, wk, pk in items:
        if wk <= libre:                             # cabe entero
            val += pk
            libre -= wk
        elif libre > 0:                             # cabe fracción
            val += ratio * libre
            libre = 0
    return val


# ===================Busqueda por anchura para Knapsack.===================
def _knapsack_bfs(datos, config):
    w, p, cap = datos['Weights'], datos['Prices'], datos['Capacity']
    n = len(w)
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()

    # Frontera: (índice, peso acumulado, valor acumulado, selección)
    fr = deque([(0, 0, 0, [])])
    exp = gen = 1
    fmax = 1
    mejor = (0, [])   # (mejor valor o selección)

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)
        i, peso, val, sel = fr.popleft()
        exp += 1
        if i == n:                                  
            if val > mejor[0]:
                mejor = (val, sel)
            continue

        # Rama 1: NO tomar el ítem i
        gen += 1
        fr.append((i + 1, peso, val, sel))

        # Rama 2: SÍ tomar el ítem i (solo si cabe)
        if peso + w[i] <= cap:
            gen += 1
            fr.append((i + 1, peso + w[i], val + p[i], sel + [i]))
        fmax = max(fmax, len(fr))
    return _res(valor_objetivo=int(mejor[0]),
                costo=int(-mejor[0]),      # costo = -valor (minimizar costo = maximizar valor)
                expandidos=exp, generados=gen, frontera_max=fmax, profundidad=n,
                visual={'seleccion': mejor[1]})


# ==============Busqueda de profundidad para Knapsack==================
def _knapsack_dfs(datos, config):
    w, p, cap = datos['Weights'], datos['Prices'], datos['Capacity']
    n = len(w)
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()

    fr = [(0, 0, 0, [])]
    exp = gen = 0
    fmax = 1
    mejor = (0, [])

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)
        i, peso, val, sel = fr.pop()                
        exp += 1
        if i == n:
            if val > mejor[0]:
                mejor = (val, sel)
            continue

      # Se empuja primero el de no para que al tomar quede arriba y sea lo primero en explorar.
        gen += 1
        fr.append((i + 1, peso, val, sel))
        if peso + w[i] <= cap:
            gen += 1
            fr.append((i + 1, peso + w[i], val + p[i], sel + [i]))
        fmax = max(fmax, len(fr))
    return _res(valor_objetivo=int(mejor[0]), costo=int(-mejor[0]),
                expandidos=exp, generados=gen, frontera_max=fmax, profundidad=n,
                visual={'seleccion': mejor[1]})

#===================Busqueda de costo uniforme para Knapsack.=======================
def _knapsack_ucs(datos, config):
    w, p, cap = datos['Weights'], datos['Prices'], datos['Capacity']
    n = len(w)
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()

    # Frontera: (costo, índice, peso, valor, selección)
    fr = [(0, 0, 0, 0, [])]
    exp = gen = 0
    fmax = 1
    mejor = (-1, [])                                # -1 indica nada

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)
        neg_val, i, peso, val, sel = heapq.heappop(fr)
        exp += 1
        if i == n:
            if val > mejor[0]:
                mejor = (val, sel)
            continue

        # No tomar
        gen += 1
        heapq.heappush(fr, (-val, i + 1, peso, val, sel))

        # Tomar (si cabe)
        if peso + w[i] <= cap:
            gen += 1
            heapq.heappush(fr, (-(val + p[i]), i + 1, peso + w[i], val + p[i], sel + [i]))
        fmax = max(fmax, len(fr))
    return _res(valor_objetivo=int(mejor[0]), costo=int(-mejor[0]),
                expandidos=exp, generados=gen, frontera_max=fmax, profundidad=n,
                visual={'seleccion': mejor[1]})


#================Heuristica para Knapsack.==============
def _knapsack_astar(datos, config):
    w, p, cap = datos['Weights'], datos['Prices'], datos['Capacity']
    n = len(w)
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()
    h0 = _h_knapsack(datos, 0, cap)                
    # Frontera: (f, g=-val, índice, peso, valor, selección)
    fr = [(-h0, 0, 0, 0, 0, [])]
    exp = gen = 0
    fmax = 1
    mejor = (-1, [])

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)
        _, g, i, peso, val, sel = heapq.heappop(fr)
        exp += 1
        if i == n:
            if val > mejor[0]:
                mejor = (val, sel)
            continue

        # No tomar h se recalcula con el mismo espacio restante
        h1 = _h_knapsack(datos, i + 1, cap - peso)
        gen += 1
        heapq.heappush(fr, (-(val + h1), -val, i + 1, peso, val, sel))

        # Tomarh se recalcula con menos espacio 
        if peso + w[i] <= cap:
            h2 = _h_knapsack(datos, i + 1, cap - peso - w[i])
            gen += 1
            heapq.heappush(fr, (-(val + p[i] + h2), -(val + p[i]), i + 1,
                                peso + w[i], val + p[i], sel + [i]))
        fmax = max(fmax, len(fr))
    return _res(valor_objetivo=int(mejor[0]), costo=int(-mejor[0]),
                expandidos=exp, generados=gen, frontera_max=fmax, profundidad=n,
                visual={'seleccion': mejor[1]})
