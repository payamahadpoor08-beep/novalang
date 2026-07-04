local W,H,M=600,600,200; local total=0
for py=0,H-1 do for px=0,W-1 do
 local x0=px/W*3.5-2.5; local y0=py/H*2.0-1.0; local x,y,it=0.0,0.0,0
 while x*x+y*y<=4.0 and it<M do local xt=x*x-y*y+x0; y=2.0*x*y+y0; x=xt; it=it+1 end
 if it==M then total=total+1 end
end end
print(total)
