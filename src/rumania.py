# ============================ RUMANIA ================================
def _grafo_rumania(datos):
    g = {}
    for a, vec in datos['ROADS'].items():
        g.setdefault(a, {})                     # asegura que 'a' exista
        for b, d in vec.items():
            g.setdefault(b, {})                 # asegura que 'b' exista
            g[a][b] = d                         # dirección a → b
            g[b][a] = d                         # dirección b → a 
    return g

#Recibe el destino y devuelve una función h(ciudad) que calcula la distancia usando A*.
def _h_rumania(datos, destino):
    loc = datos['LOCATIONS']                    # {A: (91,492)...}
    xd, yd = loc[destino]                       # coordenadas del destino Bucarest
    def h(c):
        xc, yc = loc[c]                         # coordenadas de la ciudad actual
        # Fórmula de distancia euclidiana: √((x1−x2)² + (y1−y2)²)
        return ((xc - xd) ** 2 + (yc - yd) ** 2) ** 0.5
    return h

#============ Busqueda por anchura para Rumania==========================
# Explora por niveles y ve el camino con menos aristas,pero NO el de menor costo.

def _rumania_bfs(datos, config):
    g = _grafo_rumania(datos)                                 
    origen, destino = config['origen'], config['destino']       
    lim_exp = config.get('limite_expandidos', 10**6)            # corte por nodos
    lim_s   = config.get('limite_s', 10)                        # corte por tiempo
    t0 = time.perf_counter()                                    # timepo

    # Cada elemento es (ciudad, camino, costo acumulado)
    fr = deque([(origen, [origen], 0)])
    visit = {origen}                                          # ciudades ya vistas
    exp = gen = 1                                             # contador
    fmax = 1                                                  # tamaño max de la frontera

    while fr:
        # Verifica si se agotó algún límite antes de expandir
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)

        # Saca el primero de la cola 
        ciudad, camino, costo = fr.popleft()
        exp += 1                                                

        # ¿Llegamos al destino?
        if ciudad == destino:
            return _res(costo=costo, expandidos=exp, generados=gen,
                        frontera_max=fmax, profundidad=len(camino)-1,
                        visual={'ruta': camino})

        # Genera sucesores ordenados alfabéticamente 
        for v in sorted(g[ciudad]):
            if v not in visit:         # evita ciclos
                visit.add(v)
                gen += 1               # contamos
                fr.append((v, camino + [v], costo + g[ciudad][v]))
        fmax = max(fmax, len(fr))                              
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)

# ================Busqueda de profundidad para Rumania=================
def _rumania_dfs(datos, config):
    g = _grafo_rumania(datos)
    origen, destino = config['origen'], config['destino']
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()

    # Pila: cada elemento lleva (ciudad, camino, costo, set de visitados en la rama)
    fr = [(origen, [origen], 0, {origen})]
    exp = gen = 0
    fmax = 1

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)

        # Saca el último 
        ciudad, camino, costo, visto = fr.pop()
        exp += 1
        if ciudad == destino:
            return _res(costo=costo, expandidos=exp, generados=gen,
                        frontera_max=fmax, profundidad=len(camino)-1,
                        visual={'ruta': camino})

        # Se empuja en orden inverso para que el primero  salga primero
        for v in sorted(g[ciudad], reverse=True):
            if v not in visto:
                gen += 1
                fr.append((v, camino + [v], costo + g[ciudad][v], visto | {v}))
        fmax = max(fmax, len(fr))
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)

# ================Busqueda de costo uniforme para Rumania====================
#Garantiza la ruta más barata cuando todos los costos son positivos.
def _rumania_ucs(datos, config):
    g = _grafo_rumania(datos)
    origen, destino = config['origen'], config['destino']
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()
    fr = [(0, 0, origen, [origen])]
    mejor = {origen: 0}                  # mejor costo conocido por ciudad
    cont = 0                            
    exp = gen = 0
    fmax = 1

    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)

        # Saca el de menor costo acumulado
        costo, _, ciudad, camino = heapq.heappop(fr)
        exp += 1
        if ciudad == destino:
            return _res(costo=costo, expandidos=exp, generados=gen,
                        frontera_max=fmax, profundidad=len(camino)-1,
                        visual={'ruta': camino})
        for v, d in g[ciudad].items():
            ng = costo + d        # nuevo costo acumulado
            # Solo expande si mejora el mejor costo conocido de v
            if v not in mejor or ng < mejor[v]:
                mejor[v] = ng
                cont += 1
                gen += 1
                heapq.heappush(fr, (ng, cont, v, camino + [v]))
        fmax = max(fmax, len(fr))
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)

# ==================Heuristica(A*) para Rumania======================
def _rumania_astar(datos, config):
    g = _grafo_rumania(datos)
    h = _h_rumania(datos, config['destino'])        # función A
    origen, destino = config['origen'], config['destino']
    lim_exp = config.get('limite_expandidos', 10**6)
    lim_s   = config.get('limite_s', 10)
    t0 = time.perf_counter()

    # Frontera ordenada por f = g + h. Elemento = (f, g, contador, ciudad, camino)
    fr = [(h(origen), 0, 0, origen, [origen])]
    mejor = {origen: 0}
    cont = 0
    exp = gen = 0
    fmax = 1
    while fr:
        if exp > lim_exp or time.perf_counter() - t0 > lim_s:
            return _res('limite', expandidos=exp, generados=gen, frontera_max=fmax)

        # Saca el de menor f
        _, costo, _, ciudad, camino = heapq.heappop(fr)
        exp += 1
        if ciudad == destino:
            return _res(costo=costo, expandidos=exp, generados=gen,
                        frontera_max=fmax, profundidad=len(camino)-1,
                        visual={'ruta': camino})
            
        for v, d in g[ciudad].items():
            ng = costo + d  
            if v not in mejor or ng < mejor[v]:
                mejor[v] = ng
                cont += 1
                gen += 1
                # Prioridad = g + h
                heapq.heappush(fr, (ng + h(v), ng, cont, v, camino + [v]))
        fmax = max(fmax, len(fr))
    return _res('sin_solucion', expandidos=exp, generados=gen, frontera_max=fmax)
