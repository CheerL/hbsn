import numpy as np


def geodesicwelding(a, b, s, t):
    with np.errstate(divide='ignore', invalid='ignore'):
        n = len(s)
        z = np.concatenate((s, a, [0]))
        w = np.concatenate((t, b, [np.inf]))
        z = step0(z, z[0], z[1])
        w = step0(w, w[0], w[1])
        for i in range(2, n):
            z = step1(z, z[i], 1j)
            w = step1(w, w[i], -1j)
        z = step2(z, z[0])
        w = step2(w, w[0])
        for i in range(n-2, 0, -1):
            c1 = z[i]
            c2 = w[i]
            z = step3(z, c1, c2)
            w = step3(w, c1, c2)
        z = step4(z, z[0])
        w = step4(w, w[0])
        a = step5(z[n:-1], z[-1], w[-1])
        b = step5(w[n:-1], z[-1], w[-1])
        return a, b

def step0(z, p, q):
    w = np.sqrt((z-q)/(z-p))
    w[np.isinf(z)] = 1
    return w

def step1(z, p, m):
    c = np.real(p) / np.abs(p)**2
    d = np.imag(p) / np.abs(p)**2
    t = c * z / (1 + 1j * d * z)
    t[np.isinf(z)] = c / (1j * d)
    w = np.sqrt(t**2 - 1)
    k = np.imag(w) * np.imag(t) < 0
    w[k] = -w[k]
    w[z == 0] = m
    w[z == p] = 0
    return w

def step2(z, p):
    w = z / (1 - z / p)
    w[np.isinf(z)] = -p
    return w

def step3(z, p, q, eps=1e-10):
    p = np.imag(p)
    q = np.imag(q)
    p = p if p else eps
    q = q if q else eps
    a = -2 * p * q / (p - q)
    b = (p + q) / (p - q)
    r = z / (a - 1j * b * z)
    r[np.isnan(z)] = 1j / b
    r[np.isinf(z)] = 1j / b
    w = np.sqrt(r**2 + 1)
    s = np.imag(r) * np.imag(w) < 0
    w[s] = -w[s]
    return w

def step4(z, p):
    w = (z / (1 - z / p))**2
    w[np.isinf(z)] = p**2
    w[np.isnan(w)] = np.inf
    return w

def step5(z, p, q):
    w = (z - p) / (z - q)
    w[np.isinf(z)] = 1
    return w