#include <stdio.h>
#include <stdint.h>
static uint32_t K[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define R(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void sha256(const unsigned char*m,int ml,uint32_t*out){
  unsigned char b[128]; int i; for(i=0;i<ml;i++)b[i]=m[i]; b[ml]=0x80; int n=ml+1;
  while(n%64!=56)b[n++]=0; long bits=(long)ml*8; for(i=0;i<8;i++)b[n++]=(bits>>((7-i)*8))&255;
  uint32_t h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a,h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
  for(int bl=0;bl<n/64;bl++){uint32_t w[64];
    for(int t=0;t<16;t++){int o=bl*64+t*4; w[t]=(b[o]<<24)|(b[o+1]<<16)|(b[o+2]<<8)|b[o+3];}
    for(int t=16;t<64;t++){uint32_t s0=R(w[t-15],7)^R(w[t-15],18)^(w[t-15]>>3),s1=R(w[t-2],17)^R(w[t-2],19)^(w[t-2]>>10);w[t]=w[t-16]+s0+w[t-7]+s1;}
    uint32_t a=h0,bb=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
    for(int t=0;t<64;t++){uint32_t S1=R(e,6)^R(e,11)^R(e,25),ch=(e&f)^(~e&g),t1=hh+S1+ch+K[t]+w[t],S0=R(a,2)^R(a,13)^R(a,22),mj=(a&bb)^(a&c)^(bb&c),t2=S0+mj;hh=g;g=f;f=e;e=d+t1;d=c;c=bb;bb=a;a=t1+t2;}
    h0+=a;h1+=bb;h2+=c;h3+=d;h4+=e;h5+=f;h6+=g;h7+=hh;}
  out[0]=h0;out[1]=h1;out[2]=h2;out[3]=h3;out[4]=h4;out[5]=h5;out[6]=h6;out[7]=h7;
}
static void mine(uint32_t*prev,int idx,int diff,uint32_t*out){
  for(uint32_t nonce=0;;nonce++){unsigned char b[40];int p=0,i;
    for(i=0;i<8;i++){b[p++]=(prev[i]>>24)&255;b[p++]=(prev[i]>>16)&255;b[p++]=(prev[i]>>8)&255;b[p++]=prev[i]&255;}
    b[p++]=(idx>>24)&255;b[p++]=(idx>>16)&255;b[p++]=(idx>>8)&255;b[p++]=idx&255;
    b[p++]=(nonce>>24)&255;b[p++]=(nonce>>16)&255;b[p++]=(nonce>>8)&255;b[p++]=nonce&255;
    sha256(b,40,out); if((out[0]>>(32-diff*4))==0)return;}
}
int main(){int diff=3,nblocks=20;uint32_t prev[8]={0,0,0,0,0,0,0,0},h[8];
  for(int i=0;i<nblocks;i++){mine(prev,i,diff,h);for(int j=0;j<8;j++)prev[j]=h[j];}
  for(int i=0;i<8;i++)printf("%08x",prev[i]);printf("\n");return 0;}
