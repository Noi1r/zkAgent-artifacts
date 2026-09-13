

def verify_factorization(n: int, p: int, q: int) -> bool:

    if p <= 1:
        return False
    if q <= 1:
        return False
    if p * q != n:
        return False
    return True

def is_prime_simple(n: int) -> bool:

    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i = i + 2
    return True
