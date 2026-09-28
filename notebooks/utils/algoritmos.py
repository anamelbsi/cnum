import math

import numpy as np


def _approx_fprime(x, f):
    """Aproxima a derivada por diferenças finitas sem depender de SciPy."""
    x = np.asarray(x, dtype=float)
    if x.size != 1:
        raise ValueError("_approx_fprime espera um escalar.")
    h = 1e-8 * max(1.0, abs(float(x)))
    return (f(np.array([x[0] + h]))[0] - f(np.array([x[0] - h]))[0]) / (2 * h)


def bissecao(f,     # função que queremos encontrar a raiz
              a,    # a início do intervalo
              b,    # b fim do intervalo
              TOL,   # erro tolerado
              iter=16):  # número máximo de iterações
    c = (a + b) / 2  # ponto médio entre os valores a e b
    if f(a) * f(b) > 0:
        raise ValueError("Nenhuma raiz encontrada no intervalo.")
    else:
        i = 0  # variável contador
        ERRO = abs(f(b) - f(a))  # diferença entre os valores de y

        while ERRO > TOL and i < iter:  # loop iterativo com parada
            c = (a + b) / 2.0
            if f(c) == 0:
                return c, i
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
            i += 1
            ERRO = abs(f(b) - f(a))
        return c, i


def pontofixo(a, g, TOL=1e-8, max_iter=10000):
    """Implementa o método de iteração de ponto fixo."""
    x = g(a)
    for _ in range(max_iter):
        if abs(x - a) <= TOL:
            return x
        a = x
        x = g(a)
    return x


def newton_raphson(a, f, TOL=1e-8, df=None, max_iter=10000):
    """Implementa o método de Newton-Raphson.

    Se df não for informado, usa uma aproximação numérica da derivada via
    diferenças finitas.
    """
    if df is None:
        def dfn(x):
            return _approx_fprime(np.array([x]), lambda v: np.array([f(v[0])]))[0]
    else:
        dfn = df

    g = lambda x: x - f(x) / dfn(x)
    return pontofixo(a, g, TOL=TOL, max_iter=max_iter)


def secante(a, b, f, TOL=1e-8, max_iter=10000):
    """Implementa o método da secante para a raiz de f(x) = 0."""
    fa = f(a)
    fb = f(b)

    if abs(fa) <= TOL:
        return a
    if abs(fb) <= TOL:
        return b

    for _ in range(max_iter):
        if abs(fb) <= TOL:
            return b
        if abs(fb - fa) <= 1e-30:
            return b

        x = b - fb * (b - a) / (fb - fa)
        if abs(x - b) <= TOL:
            return x

        a, b = b, x
        fa, fb = fb, f(b)

    return b