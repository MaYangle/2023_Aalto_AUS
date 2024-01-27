function dzdt = odefcn(t, z, m, b, k, zk, theta1, theta2, g, i)
    current = i(t);
    dzdt = zeros(2, 1);
    dzdt(1) = z(2);
    dzdt(2) = (k / m) * (zk - z(1)) - (b / m) * z(2) - g - (theta1 / m) * ((current / (z(1) + theta2))^2);
end
