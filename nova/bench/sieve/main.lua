local n=2000000; local s={}; for i=0,n-1 do s[i]=1 end; s[0]=0; s[1]=0
local i=2; while i*i<n do if s[i]==1 then local j=i*i; while j<n do s[j]=0; j=j+i end end; i=i+1 end
local c=0; for k=0,n-1 do c=c+s[k] end; print(c)
