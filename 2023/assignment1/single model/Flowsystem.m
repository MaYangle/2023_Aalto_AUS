%% Ljung 1994 case 3: flow system
u = 1;      
A = 1;      
g = 9.8;    
a = 0.2;

model = @(t,x) -a*sqrt(2*g)/A*sqrt(x(1)) + 1/A*u+0.05;
hfunc = @(x) x;

options = odeset('RelTol',1e-4,'AbsTol',1e-6);
timeSpan = [0 20];
initCond = 0;
[T1,X1] = ode45(model,timeSpan, initCond,options);
initCond = 2;
[T2,X2] = ode45(model,timeSpan, initCond,options);

Y1 = hfunc(X1); % we are only interested in the height
Y2 = hfunc(X2);

clf
plot(T1,Y1,'r-.'); hold on; plot(T2,Y2,'b-');
title('Solution of the liquid level of a water tank');
xlabel('time');
ylabel('x');
legend('x(0)=0','x(0)=2')

%% if we are interested in the outflow
hfunc = @(x) a * sqrt(2*g)* sqrt(x);

Y1 = hfunc(X1);
Y2 = hfunc(X2);

clf
plot(T1,Y1,'r-.'); hold on; plot(T2,Y2,'b-');
title('Solution of the output folow of a water tank');
xlabel('time');
ylabel('x');
legend('x(0)=0','x(0)=2')

%% using q as the internal state

model = @(t,x) -a^2*g/A + a^2*g/A*x(1)^-1*u;
hfunc = @(x) 1/(2*a^2*g)*x.^2;

options = odeset('RelTol',1e-4,'AbsTol',1e-6);
timeSpan = [0 20];
initCond = 0;
[T1,X1] = ode45(model,timeSpan, initCond,options);
initCond = 2;
[T2,X2] = ode45(model,timeSpan, initCond,options);

Y1 = hfunc(X1); % we are only interested in the height
Y2 = hfunc(X2);

clf
plot(T1,Y1,'r-.'); hold on; plot(T2,Y2,'b-');
title('Solution of the liquid level of a water tank');
xlabel('time');
ylabel('x');
legend('x(0)=0','x(0)=2')
















%% Linearized transfer function

num = 1/A;
den = [1 +sqrt(2)*u^2/(4*sqrt(g)*A*a)];
sys = tf(num, den);
L0 = 1/(2*a^2*g)*u^2;

T = 0:0.01:10;
T = T';
U = ones(size(T))-u;
Y = lsim(sys,U+0.05,T,0);

plot(T,Y+L0)
legend('L(0)=0','L(0)=2','linearized, L(0) = 1')

figure, bode(sys);