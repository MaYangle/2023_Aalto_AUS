%% Mass-Spring-Damper System
clear all;
clf;
k = 1;
m = 2;
b = 3;
    
A = [ 0 1; -k/m -b/m];
B = [ 0 1/m]';
C = [1 0];
D = 0;

u = 1;
system = @(t,x) A*x + B*u;
output = @(x) C*x + D*u;

options = odeset('RelTol',1e-4,'AbsTol',1e-6);

timeSpan = [0 20];
initCond = [0 0]';
[T1,X1] = ode45(system,timeSpan, initCond,options);
Y1 = output(X1');
 
u = @(t) sin(t);
system = @(t,x) A*x + B*u(t); % u(t) = sin(t)
output = @(t,x) C*x + D*u(t);
[T2,X2] = ode45(system,timeSpan,initCond,options);
Y2 = output(T2', X2');

plot(T1,Y1,'r-.'); hold on; plot(T2,Y2,'b-');
title('Solution of a mass-spring-damper system');
xlabel('time');
ylabel('y(t)');
legend('u = 1','u = sin(t)')

%%