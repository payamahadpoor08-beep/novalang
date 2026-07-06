w, h, maxi = 600, 600, 200
total = 0
(0...h).each do |py|
  (0...w).each do |px|
    x0 = px.to_f/w*3.5 - 2.5
    y0 = py.to_f/h*2.0 - 1.0
    x = 0.0; y = 0.0; it = 0
    while x*x + y*y <= 4.0 && it < maxi
      xt = x*x - y*y + x0; y = 2.0*x*y + y0; x = xt; it += 1
    end
    total += 1 if it == maxi
  end
end
puts total
