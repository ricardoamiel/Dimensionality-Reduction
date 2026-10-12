# -*- coding: utf-8 -*-
"""
Implementaciones sencillas de reducción de dimensionalidad, solo con NumPy.

    pca(X, k)            PCA por eigen descomposición de la matriz de covarianza
    svd(A, k)            SVD truncada por iteración de potencia con deflación
    pca_via_svd(X, k)    PCA calculado a partir de la SVD de los datos centrados
    tsne(X, ...)         t-SNE exacto (van der Maaten y Hinton, 2008)
    umap(X, ...)         UMAP simplificado (McInnes et al., 2018)

Ninguna función llama a np.linalg.svd ni a np.linalg.eig: el objetivo es que
cada paso del algoritmo quede a la vista. Las versiones de NumPy y scikit-learn
solo se usan en el notebook para comprobar los resultados.
"""
import numpy as np


# ======================================================================
# Bloque común: autovectores por iteración de potencia
# ======================================================================
def iteracion_potencia(M, iters=1000, tol=1e-10, semilla=0):
    """Autovector dominante de una matriz simétrica semidefinida positiva.

    Se multiplica un vector al azar por M una y otra vez; la componente en la
    dirección del autovalor más grande crece más rápido que las demás y al
    normalizar en cada paso el vector converge a esa dirección.
    """
    rng = np.random.default_rng(semilla)
    v = rng.normal(size=M.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = M @ v
        norma = np.linalg.norm(w)
        if norma == 0:
            return 0.0, v
        w /= norma
        if np.linalg.norm(w - v) < tol:
            v = w
            break
        v = w
    return float(v @ M @ v), v          # cociente de Rayleigh y autovector


def autovectores(M, k, **kw):
    """Los k autovectores principales de M por potencia con deflación.

    Tras encontrar (lambda, v) se resta lambda * v v^T de M; así la siguiente
    iteración de potencia ya no ve esa dirección y encuentra la siguiente.
    """
    M = np.array(M, dtype=float)
    valores, vectores = [], []
    for i in range(k):
        lam, v = iteracion_potencia(M, semilla=i, **kw)
        if v[np.argmax(np.abs(v))] < 0:  # convención de signo estable
            v = -v
        valores.append(lam)
        vectores.append(v)
        M = M - lam * np.outer(v, v)
    return np.array(valores), np.column_stack(vectores)


# ======================================================================
# PCA
# ======================================================================
def pca(X, k):
    """PCA clásico en cinco pasos.

    1. Centrar: restar la media de cada variable.
    2. Covarianza: C = Xc^T Xc / (n - 1).
    3. Autovectores de C ordenados por autovalor (las componentes).
    4. Proyectar: Z = Xc W.
    5. Varianza explicada: lambda_i / traza(C).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    media = X.mean(axis=0)
    Xc = X - media
    C = Xc.T @ Xc / (n - 1)
    valores, W = autovectores(C, k)
    Z = Xc @ W
    return {
        "media": media,
        "covarianza": C,
        "componentes": W,                       # columnas = direcciones
        "varianzas": valores,                   # autovalores
        "varianza_explicada": valores / np.trace(C),
        "Z": Z,                                 # datos en el nuevo espacio
        "reconstruccion": Z @ W.T + media,
    }


# ======================================================================
# SVD
# ======================================================================
def svd(A, k):
    """SVD truncada A ~ U_k S_k V_k^T.

    1. Los vectores singulares derechos v_i son autovectores de A^T A.
    2. Los valores singulares son sigma_i = raíz(lambda_i).
    3. Los vectores singulares izquierdos salen de u_i = A v_i / sigma_i.
    """
    A = np.asarray(A, dtype=float)
    lam, V = autovectores(A.T @ A, k)
    s = np.sqrt(np.clip(lam, 0, None))
    U = np.zeros((A.shape[0], k))
    for i in range(k):
        if s[i] > 1e-12:
            U[:, i] = A @ V[:, i] / s[i]
    return U, s, V.T


def reconstruir(U, s, Vt, k=None):
    """Aproximación de rango k: suma de las k capas sigma_i u_i v_i^T."""
    k = len(s) if k is None else k
    return (U[:, :k] * s[:k]) @ Vt[:k]


def pca_via_svd(X, k):
    """El mismo PCA, pero desde la SVD de los datos centrados.

    Si Xc = U S V^T entonces C = V (S^2 / (n - 1)) V^T: las componentes son
    las filas de V^T y las varianzas son sigma^2 / (n - 1).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    media = X.mean(axis=0)
    U, s, Vt = svd(X - media, k)
    varianzas = s ** 2 / (n - 1)
    total = ((X - media) ** 2).sum() / (n - 1)
    return {"media": media, "componentes": Vt.T, "varianzas": varianzas,
            "varianza_explicada": varianzas / total, "Z": U * s}


# ======================================================================
# t-SNE
# ======================================================================
def _distancias2(X):
    s = (X ** 2).sum(1)
    D = s[:, None] + s[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def _afinidades_tsne(D2, perplejidad, tol=1e-5, iters=60):
    """P condicional con una sigma por punto elegida por búsqueda binaria
    para que la entropía de P_i valga log2(perplejidad)."""
    n = D2.shape[0]
    P = np.zeros((n, n))
    objetivo = np.log(perplejidad)
    for i in range(n):
        d = np.delete(D2[i], i)
        beta, lo, hi = 1.0, 0.0, np.inf          # beta = 1 / (2 sigma^2)
        for _ in range(iters):
            p = np.exp(-(d - d.min()) * beta)
            p /= p.sum()
            H = -(p * np.log(p + 1e-12)).sum()
            if abs(H - objetivo) < tol:
                break
            if H > objetivo:                     # muy plana: subir beta
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    return P


def tsne(X, dim=2, perplejidad=30.0, iters=1000, eta=200.0, exageracion=12.0,
         semilla=0, historial=False):
    """t-SNE exacto, O(n^2) por iteración; pensado para n de unos pocos miles.

    1. Afinidades P en el espacio original (gaussianas con perplejidad fija).
    2. Simetrizar: P = (P + P^T) / 2n.
    3. Afinidades Q en el mapa con una t de Student de un grado de libertad.
    4. Descenso de gradiente sobre KL(P || Q) con momento.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    P = _afinidades_tsne(_distancias2(X), perplejidad)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    rng = np.random.default_rng(semilla)
    Y = rng.normal(scale=1e-4, size=(n, dim))
    vel = np.zeros_like(Y)
    pasos = []
    for t in range(iters):
        exa = exageracion if t < 250 else 1.0
        mom = 0.5 if t < 250 else 0.8
        num = 1.0 / (1.0 + _distancias2(Y))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (exa * P - Q) * num
        grad = 4 * (np.diag(PQ.sum(1)) - PQ) @ Y
        vel = mom * vel - eta * grad
        Y = Y + vel
        Y -= Y.mean(0)
        if historial and t % 10 == 0:
            pasos.append(Y.copy())
    kl = float((P * np.log(P / Q)).sum())
    return (Y, kl, pasos) if historial else (Y, kl)


# ======================================================================
# UMAP (versión didáctica)
# ======================================================================
def _ajustar_ab(min_dist, spread=1.0):
    """Ajusta a y b de 1 / (1 + a d^(2b)) a la curva objetivo de UMAP
    por mínimos cuadrados en una rejilla (Gauss Newton muy simple)."""
    d = np.linspace(1e-3, 3 * spread, 300)
    y = np.where(d < min_dist, 1.0, np.exp(-(d - min_dist) / spread))
    a, b = 1.6, 0.9
    for _ in range(200):
        f = 1 / (1 + a * d ** (2 * b))
        r = f - y
        dfa = -d ** (2 * b) * f ** 2
        dfb = -a * 2 * np.log(d) * d ** (2 * b) * f ** 2
        J = np.column_stack([dfa, dfb])
        paso = np.linalg.lstsq(J, -r, rcond=None)[0]
        a, b = a + paso[0], b + paso[1]
    return a, b


def umap(X, dim=2, vecinos=15, min_dist=0.1, epocas=300, negativos=5,
         semilla=0, devolver_grafo=False):
    """UMAP simplificado en cuatro pasos.

    1. Grafo kNN exacto.
    2. Pesos difusos w_ij = exp(-(d_ij - rho_i) / sigma_i), con rho_i la
       distancia al vecino más cercano y sigma_i tal que la suma de pesos
       valga log2(k).
    3. Unión difusa: w = w_ij + w_ji - w_ij w_ji.
    4. Optimizar la entropía cruzada con descenso estocástico: las aristas
       atraen y muestras negativas al azar repelen.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(semilla)
    D = np.sqrt(_distancias2(X))
    idx = np.argsort(D, axis=1)[:, 1:vecinos + 1]
    dist = np.take_along_axis(D, idx, axis=1)
    rho = dist[:, 0]
    objetivo = np.log2(vecinos)
    W = np.zeros((n, n))
    for i in range(n):
        lo, hi, s = 0.0, np.inf, 1.0
        for _ in range(64):
            suma = np.exp(-np.maximum(dist[i] - rho[i], 0) / s).sum()
            if abs(suma - objetivo) < 1e-5:
                break
            if suma > objetivo:
                hi = s
                s = (lo + hi) / 2
            else:
                lo = s
                s = s * 2 if hi == np.inf else (lo + hi) / 2
        W[i, idx[i]] = np.exp(-np.maximum(dist[i] - rho[i], 0) / s)
    W = W + W.T - W * W.T
    filas, cols = np.nonzero(np.triu(W))
    pesos = W[filas, cols]

    a, b = _ajustar_ab(min_dist)
    # inicialización con PCA escalado a una caja de lado 20
    Y = pca(X, dim)["Z"]
    Y = 10 * Y / np.abs(Y).max()
    prob = pesos / pesos.max()
    for e in range(epocas):
        alfa = 1.0 - e / epocas
        activas = rng.random(len(pesos)) < prob
        i, j = filas[activas], cols[activas]
        # atracción a lo largo de las aristas del grafo
        dif = Y[i] - Y[j]
        d2 = (dif ** 2).sum(1) + 1e-6
        coef = (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b)
        g = np.clip(coef[:, None] * dif, -4, 4) * alfa
        np.add.at(Y, i, g)
        np.add.at(Y, j, -g)
        # repulsión con muestras negativas
        ii = np.repeat(i, negativos)
        jj = rng.integers(0, n, size=len(ii))
        dif = Y[ii] - Y[jj]
        d2 = (dif ** 2).sum(1) + 1e-3
        coef = (2 * b) / (d2 * (1 + a * d2 ** b))
        g = np.clip(coef[:, None] * dif, -4, 4) * alfa
        np.add.at(Y, ii, g)
    if devolver_grafo:
        return Y, (filas, cols, pesos)
    return Y
