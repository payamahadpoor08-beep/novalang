n = 2000000
s = Array.new(n, 1)
s[0] = 0; s[1] = 0
i = 2
while i*i < n
  if s[i] == 1
    j = i*i
    while j < n; s[j] = 0; j += i; end
  end
  i += 1
end
puts s.sum
