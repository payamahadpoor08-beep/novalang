K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

MASK = 0xffffffff


def rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & MASK


def sha256(msg):
    h = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
    ml = len(msg)
    m = list(msg)
    m.append(128)
    while len(m) % 64 != 56:
        m.append(0)
    bits = ml * 8
    for i in range(8):
        m.append((bits >> ((7 - i) * 8)) & 255)
    nblocks = len(m) // 64
    for b in range(nblocks):
        w = [0] * 64
        for t in range(16):
            o = b * 64 + t * 4
            w[t] = ((m[o] << 24) | (m[o + 1] << 16) | (m[o + 2] << 8) | m[o + 3]) & MASK
        for t in range(16, 64):
            s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
            s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
            w[t] = (w[t - 16] + s0 + w[t - 7] + s1) & MASK
        a, b2, c, d, e, f, g, hh = h
        for t in range(64):
            S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
            ch = (e & f) ^ ((e ^ MASK) & g)
            t1 = (hh + S1 + ch + K[t] + w[t]) & MASK
            S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
            mj = (a & b2) ^ (a & c) ^ (b2 & c)
            t2 = (S0 + mj) & MASK
            hh, g, f, e, d, c, b2, a = g, f, e, (d + t1) & MASK, c, b2, a, (t1 + t2) & MASK
        h[0] = (h[0] + a) & MASK
        h[1] = (h[1] + b2) & MASK
        h[2] = (h[2] + c) & MASK
        h[3] = (h[3] + d) & MASK
        h[4] = (h[4] + e) & MASK
        h[5] = (h[5] + f) & MASK
        h[6] = (h[6] + g) & MASK
        h[7] = (h[7] + hh) & MASK
    return h


def hexdigest(h):
    return "".join("%08x" % x for x in h)


def wbytes(arr, x):
    for i in range(4):
        arr.append((x >> ((3 - i) * 8)) & 255)


def header(prev, index, nonce):
    b = []
    for i in range(8):
        wbytes(b, prev[i])
    wbytes(b, index)
    wbytes(b, nonce)
    return b


def mine(prev, index, diff):
    nonce = 0
    while True:
        h = sha256(header(prev, index, nonce))
        if (h[0] >> (32 - diff * 4)) == 0:
            return h
        nonce += 1


def main():
    diff = 3
    nblocks = 20
    prev = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(nblocks):
        prev = mine(prev, i, diff)
    print(hexdigest(prev))


main()
