#include <stdio.h>
#include <stdlib.h>
int main(){ long n=2000000; char*s=malloc(n); for(long i=0;i<n;i++)s[i]=1; s[0]=0;s[1]=0;
 for(long i=2;i*i<n;i++) if(s[i]) for(long j=i*i;j<n;j+=i) s[j]=0;
 long c=0; for(long k=0;k<n;k++) c+=s[k]; printf("%ld\n",c); free(s); return 0; }
