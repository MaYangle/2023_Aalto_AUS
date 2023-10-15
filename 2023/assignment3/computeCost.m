function SSE = computeCost(k,t,z)
k1 = k(1);
k2 = k(2);

y1 = k1*(1-exp(k2*t));

y2 = y1 - z;

SSE = sum(y2.^2);
end

