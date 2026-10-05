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


def _jacobiano_numerico(x, F, eps=1e-8):
    """Calcula a matriz Jacobiana de F(x) por diferenças finitas."""
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)

    for j in range(n):
        h = eps * max(1.0, abs(float(x[j])))
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h

        F_plus = np.asarray(F(x_plus), dtype=float)
        F_minus = np.asarray(F(x_minus), dtype=float)
        J[:, j] = (F_plus - F_minus) / (2.0 * h)

    return J


def JN(x, F, eps=1e-8):
    """Matriz Jacobiana aproximada para um sistema F(x)=0."""
    return _jacobiano_numerico(x, F, eps=eps)


def G(x, F, J):
    """Passo de Newton para sistemas: x_{k+1} = x_k - J(x_k)^(-1) F(x_k)."""
    x = np.asarray(x, dtype=float)
    Fx = np.asarray(F(x), dtype=float)
    Jx = np.asarray(J(x), dtype=float)
    return x - np.linalg.solve(Jx, Fx)


def GN(x, F, eps=1e-8):
    """Passo de Newton com Jacobiano numericamente aproximado."""
    return G(x, F, lambda v: _jacobiano_numerico(v, F, eps=eps))


def fixed_point(a, g, TOL=1e-8, iter=1000):
    """Método iterativo de ponto fixo para sistemas vetoriais."""
    a = np.asarray(a, dtype=float)
    x = np.asarray(g(a), dtype=float)
    i = 1
    while np.linalg.norm(x - a, ord=np.inf) > TOL and i < iter:
        a = x
        x = np.asarray(g(a), dtype=float)
        i += 1
    return x


def newton_raphson_sistema(x0, F, TOL=1e-8, max_iter=100, eps=1e-8):
    """Newton-Raphson para sistemas não lineares."""
    x = np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        Fx = np.asarray(F(x), dtype=float)
        if np.linalg.norm(Fx, ord=np.inf) <= TOL:
            return x

        Jx = _jacobiano_numerico(x, F, eps=eps)
        try:
            dx = np.linalg.solve(Jx, Fx)
        except np.linalg.LinAlgError:
            dx = np.linalg.pinv(Jx) @ Fx

        x_next = x - dx
        if np.linalg.norm(x_next - x, ord=np.inf) <= TOL:
            return x_next
        x = x_next

    return x


def _resolver_atividade_2():
    def F(x):
        x1, x2, x3 = x
        return np.array(
            [
               6.0 * x1 - 2.0 * x2 + np.exp(x3) - 2.0,
               np.sin(x1) - x2 + x3,
               np.sin(x1) + 2.0 * x2 + 3.0 * x3 - 1.0,
            ],
            dtype=float,
        )

    x0 = np.array([0.0, 0.0, 0.0], dtype=float)
    raiz = newton_raphson_sistema(x0, F, TOL=1e-12)
    return raiz


def _resolver_atividade_3():
    def F(x):
        x1, x2 = x
        return np.array(
            [
               (x1 ** 2) / 8.0 + ((x2 - 1.0) ** 2) / 5.0 - 1.0,
               np.arctan(x1) + x1 - x2 - x2 ** 3,
            ],
            dtype=float,
        )

    chutes = [
        np.array([-1.2, -1.0], dtype=float),
        np.array([2.8, 1.3], dtype=float),
    ]
    raizes = [newton_raphson_sistema(chute, F, TOL=1e-12) for chute in chutes]
    return raizes


def _resolver_atividade_4():
    def C1(x):
        return 10.0 + 0.3 * x + 1e-4 * x ** 2 + 3.4e-9 * x ** 4

    def C2(x):
        return 50.0 + 0.25 * x + 2e-4 * x ** 2 + 4.3e-7 * x ** 3

    def C3(x):
        return 500.0 + 0.19 * x + 5e-4 * x ** 2 + 1.1e-7 * x ** 4

    def F(x):
        x1, x2, x3, lam = x
        return np.array(
            [
               0.3 + 2e-4 * x1 + 1.36e-8 * x1 ** 3 - lam,
               0.25 + 4e-4 * x2 + 1.29e-6 * x2 ** 2 - lam,
               0.19 + 1e-3 * x3 + 4.4e-7 * x3 ** 3 - lam,
               x1 + x2 + x3 - 1500.0,
            ],
            dtype=float,
        )

    x0 = np.array([450.0, 900.0, 150.0, 0.3], dtype=float)
    return newton_raphson_sistema(x0, F, TOL=1e-12)


if __name__ == "__main__":
    print("Atividade 2:", _resolver_atividade_2())
    print("Atividade 3:", _resolver_atividade_3())
    print("Atividade 4:", _resolver_atividade_4())