package main
import "fmt"
func main() {
	n := 2000000
	s := make([]byte, n)
	for i := range s { s[i] = 1 }
	s[0], s[1] = 0, 0
	for i := 2; i*i < n; i++ { if s[i] == 1 { for j := i * i; j < n; j += i { s[j] = 0 } } }
	c := int64(0)
	for _, v := range s { c += int64(v) }
	fmt.Println(c)
}
