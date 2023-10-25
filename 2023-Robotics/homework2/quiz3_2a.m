a = [-1.7035 4.8305 8.3666];
ax = a(1);
ay = a(2);
az = a(3);
g = 9.81;
theta_p = asin(-ax/g);
theta_r =  atan2(ay,az);

p_degree = theta_p * (180 / pi);
display(p_degree);
r_degree = theta_r * (180 / pi);
display(r_degree);
y_degree = 0.0;

R0 = rotz(0) * roty(theta_p) *rotx(theta_r);
R = R0;
R3 = eye(3);
omega_imu = [0.7 0.8 0];
for i = 1:5
    R = R + R * 0.060 *skew(omega_imu);
    R = R * R3;
    R_det = det(R);
     if i == 5
        R1 = trnorm(R);
        R1_det = det(R1);
    end
end

display(R0);
display(R);
display(R1);
display(R_det);
display(R1_det);