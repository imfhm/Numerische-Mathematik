

n = 3
K = [7, 15, 31]
def odd(n,k): return k * n + 1
for k in K:
    while n != 1:
        print(n)
        if n % 2 == 0: n /= 2
        else: n = odd(n,k)
        if n > 1000000: break
    if n <= 1000000: 
        print("{k} converges!")
        break