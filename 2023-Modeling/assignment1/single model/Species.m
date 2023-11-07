%% Ljung case 1: Species compete for the same food
clear all;
clf;
lambda = [3 2]; % birth rate
gamma = [1 1];  % mortality rate, natural
delta = [0.1 0.1];  % mortality rate, food related

mu = @ (i,N1,N2) gamma(i) + delta(i)*(N1+N2);

model1 = @(t,y) [(lambda(1)-gamma(1))*y(1) - delta(1)*(y(1)+y(2))*y(1);...
                (lambda(2)-gamma(2))*y(2) - delta(2)*(y(1)+y(2))*y(2)];
            
[t,y] = ode45(model1,[0 10],[0.1,2]);
plot(t,y(:,1),'-',t,y(:,2),'-.')
title('Solution of species compete for the same food');
xlabel('time');
ylabel('Number of individuals in thousands');
legend('species 1','species 2')

%% Ljung case 2: Predator and Prey
lambda = [1 2]; % birth rate: birth rate of pray is higher
gamma = [2 1];  % mortality rate, natural; naturally the predator is easy to die 
alpha = [1 1];  % mortality rate, food related

model2 = @(t,y) [(lambda(1)-gamma(1))*y(1) + alpha(1)*y(1)*y(2);...
                (lambda(2)-gamma(2))*y(2) - alpha(2)*y(1)*y(2)];

N1 = 2;
N2 = 2;
% N1 = (lambda(2)-gamma(2))/alpha(2);
% N2 = (gamma(1)-lambda(1))/alpha(1);

options = odeset('RelTol',1e-3,'AbsTol',[1e-6 1e-6]);
timeSpan = [0 20];
initCond = [N1 N2];
[t,y] = ode45(model2,timeSpan,initCond,options);
plot(t,y(:,1),'-',t,y(:,2),'-.')
title('Solution of predator and prey');
xlabel('time');
ylabel('Number of individuals in thousands');
legend('Predator','Pray')

%% Linearized model
lambda = [1 2];
gamma = [2 1];
alpha = [1 1];

A = [ 0 alpha(1)/alpha(2)*(lambda(2)-gamma(2)); ...
      -alpha(2)/alpha(1)*(gamma(1)-lambda(1)) 0];

model3 = @(t,y) A*y;

options = odeset('RelTol',1e-4,'AbsTol',[1e-6 1e-6]);
N1 = (lambda(2)-gamma(2))/alpha(2);
N2 = (gamma(1)-lambda(1))/alpha(1);
dN = [0.1 0];
timeSpan = [0 12];
initCond = [dN(1) dN(2)];
[t3,y3] = ode45(model3,timeSpan,initCond,options);
initCond = [N1+dN(1),N2+dN(2)];
[t2,y2] = ode45(model2,timeSpan,initCond,options);

plot(t2,y2(:,1),'-',t3,y3(:,1)+N1,'-.')
title('Solution of predator and prey');
xlabel('time');
ylabel('Number of individuals in thousands');
legend('Nonlinear','Linearized')


