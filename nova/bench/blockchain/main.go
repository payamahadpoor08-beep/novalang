package main

import "fmt"

var K = [64]uint32{
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
}

func rotr(x uint32, n uint) uint32 { return (x >> n) | (x << (32 - n)) }

func sha256(msg []byte) [8]uint32 {
	h := [8]uint32{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}
	ml := len(msg)
	m := make([]byte, 0, ml+72)
	m = append(m, msg...)
	m = append(m, 128)
	for len(m)%64 != 56 {
		m = append(m, 0)
	}
	bits := uint64(ml) * 8
	for i := 0; i < 8; i++ {
		m = append(m, byte((bits>>uint((7-i)*8))&255))
	}
	nblocks := len(m) / 64
	for b := 0; b < nblocks; b++ {
		var w [64]uint32
		for t := 0; t < 16; t++ {
			o := b*64 + t*4
			w[t] = (uint32(m[o]) << 24) | (uint32(m[o+1]) << 16) | (uint32(m[o+2]) << 8) | uint32(m[o+3])
		}
		for t := 16; t < 64; t++ {
			s0 := rotr(w[t-15], 7) ^ rotr(w[t-15], 18) ^ (w[t-15] >> 3)
			s1 := rotr(w[t-2], 17) ^ rotr(w[t-2], 19) ^ (w[t-2] >> 10)
			w[t] = w[t-16] + s0 + w[t-7] + s1
		}
		a, b2, c, d, e, f, g, hh := h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
		for t := 0; t < 64; t++ {
			S1 := rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
			ch := (e & f) ^ (^e & g)
			t1 := hh + S1 + ch + K[t] + w[t]
			S0 := rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
			mj := (a & b2) ^ (a & c) ^ (b2 & c)
			t2 := S0 + mj
			hh, g, f, e, d, c, b2, a = g, f, e, d+t1, c, b2, a, t1+t2
		}
		h[0] += a
		h[1] += b2
		h[2] += c
		h[3] += d
		h[4] += e
		h[5] += f
		h[6] += g
		h[7] += hh
	}
	return h
}

func hexdigest(h [8]uint32) string {
	const hd = "0123456789abcdef"
	s := make([]byte, 64)
	for i := 0; i < 8; i++ {
		x := h[i]
		for j := 0; j < 8; j++ {
			s[i*8+j] = hd[(x>>uint((7-j)*4))&15]
		}
	}
	return string(s)
}

func wbytes(arr []byte, x uint32) []byte {
	for i := 0; i < 4; i++ {
		arr = append(arr, byte((x>>uint((3-i)*8))&255))
	}
	return arr
}

func header(prev [8]uint32, index, nonce uint32) []byte {
	b := make([]byte, 0, 40)
	for i := 0; i < 8; i++ {
		b = wbytes(b, prev[i])
	}
	b = wbytes(b, index)
	b = wbytes(b, nonce)
	return b
}

func mine(prev [8]uint32, index, diff uint32) [8]uint32 {
	var nonce uint32 = 0
	for {
		h := sha256(header(prev, index, nonce))
		if (h[0] >> (32 - diff*4)) == 0 {
			return h
		}
		nonce++
	}
}

func main() {
	var diff uint32 = 3
	nblocks := 20
	prev := [8]uint32{0, 0, 0, 0, 0, 0, 0, 0}
	for i := 0; i < nblocks; i++ {
		prev = mine(prev, uint32(i), diff)
	}
	fmt.Println(hexdigest(prev))
}
