n=2000000; s=bytearray([1])*n; s[0]=0; s[1]=0
i=2
while i*i<n:
    if s[i]:
        s[i*i::i]=bytearray(len(s[i*i::i]))
    i+=1
print(sum(s))
