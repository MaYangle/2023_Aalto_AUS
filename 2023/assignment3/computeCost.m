function SSE = computeCost(k1,k2,x,y)

y1 = k1*(1-exp(k2*x));

y2 = y - y1;

SSE = sum(y2.^2);
end


