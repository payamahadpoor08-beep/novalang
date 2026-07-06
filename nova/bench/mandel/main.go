package main
import "fmt"
func main() {
	w, h, maxi := 600, 600, 200
	var total int64
	for py := 0; py < h; py++ { for px := 0; px < w; px++ {
		x0 := float64(px)/float64(w)*3.5 - 2.5
		y0 := float64(py)/float64(h)*2.0 - 1.0
		x, y, it := 0.0, 0.0, 0
		for x*x+y*y <= 4.0 && it < maxi { xt := x*x - y*y + x0; y = 2.0*x*y + y0; x = xt; it++ }
		if it == maxi { total++ }
	}}
	fmt.Println(total)
}
