const n=2000000; const s=new Uint8Array(n).fill(1); s[0]=0;s[1]=0;
for(let i=2;i*i<n;i++) if(s[i]) for(let j=i*i;j<n;j+=i) s[j]=0;
let c=0; for(let k=0;k<n;k++) c+=s[k]; console.log(c);
