def main():
    w, h, gens = 96, 96, 120
    n = w * h
    cur = [0] * n
    nxt = [0] * n
    seed = 1234567
    for i in range(n):
        seed = (seed * 1103515245 + 12345) & 2147483647
        cur[i] = (seed >> 16) & 1
    for _ in range(gens):
        for y in range(h):
            for x in range(w):
                c = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx != 0 or dy != 0:
                            nx = (x + dx + w) % w
                            ny = (y + dy + h) % h
                            c += cur[ny * w + nx]
                idx = y * w + x
                alive = cur[idx]
                cell = 0
                if alive == 1:
                    if c == 2 or c == 3:
                        cell = 1
                else:
                    if c == 3:
                        cell = 1
                nxt[idx] = cell
        cur[:] = nxt
    print(sum(cur))


main()
