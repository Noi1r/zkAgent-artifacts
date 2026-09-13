

def fibonacci(n: int) -> int:

    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def factorial(n: int) -> int:

    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result

def is_prime(n: int) -> bool:

    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def sum_list(numbers: list) -> int:

    total = 0
    for num in numbers:
        total = total + num
    return total
