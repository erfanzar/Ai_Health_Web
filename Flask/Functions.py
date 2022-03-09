def Reverse(Numba , min , max):
    l = min
    m = max
    out = None
    for i in range(max):
        l +=1
        m -=1
        if Numba == i:
            out = m
    return out
