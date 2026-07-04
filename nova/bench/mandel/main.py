W=H=600; M=200; total=0
for py in range(H):
    for px in range(W):
        x0=px/W*3.5-2.5; y0=py/H*2.0-1.0; x=0.0; y=0.0; it=0
        while x*x+y*y<=4.0 and it<M:
            xt=x*x-y*y+x0; y=2.0*x*y+y0; x=xt; it+=1
        if it==M: total+=1
print(total)
