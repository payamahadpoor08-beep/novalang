const W=600,H=600,M=200; let total=0;
for(let py=0;py<H;py++) for(let px=0;px<W;px++){
 let x0=px/W*3.5-2.5, y0=py/H*2.0-1.0, x=0,y=0,it=0;
 while(x*x+y*y<=4.0 && it<M){ let xt=x*x-y*y+x0; y=2.0*x*y+y0; x=xt; it++; }
 if(it===M) total++; }
console.log(total);
