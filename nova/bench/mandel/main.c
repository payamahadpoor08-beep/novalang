#include <stdio.h>
int main(){ int W=600,H=600,M=200; long total=0;
 for(int py=0;py<H;py++) for(int px=0;px<W;px++){
  double x0=(double)px/W*3.5-2.5, y0=(double)py/H*2.0-1.0, x=0,y=0; int it=0;
  while(x*x+y*y<=4.0 && it<M){ double xt=x*x-y*y+x0; y=2.0*x*y+y0; x=xt; it++; }
  if(it==M) total++; }
 printf("%ld\n",total); return 0; }
