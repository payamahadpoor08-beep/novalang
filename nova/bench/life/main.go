package main

import "fmt"

func main() {
	w, h, gens := 96, 96, 120
	n := w * h
	cur := make([]int, n)
	nxt := make([]int, n)
	var seed int64 = 1234567
	for i := 0; i < n; i++ {
		seed = (seed*1103515245 + 12345) & 2147483647
		cur[i] = int((seed >> 16) & 1)
	}
	for g := 0; g < gens; g++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := 0
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx != 0 || dy != 0 {
							nx := (x + dx + w) % w
							ny := (y + dy + h) % h
							c += cur[ny*w+nx]
						}
					}
				}
				idx := y*w + x
				alive := cur[idx]
				cell := 0
				if alive == 1 {
					if c == 2 || c == 3 {
						cell = 1
					}
				} else {
					if c == 3 {
						cell = 1
					}
				}
				nxt[idx] = cell
			}
		}
		copy(cur, nxt)
	}
	pop := 0
	for i := 0; i < n; i++ {
		pop += cur[i]
	}
	fmt.Println(pop)
}
