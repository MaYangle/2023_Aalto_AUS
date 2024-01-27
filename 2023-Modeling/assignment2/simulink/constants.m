
% Pendulum
clear
close all;
clc;
m = 0.055;
l = 0.042;
g = 9.81;
J = 1.9098e-4;
b = 3e-6;
K = 53.6e-3;
R = 9.5;
Km = K/R;

%% Reactor
clear
close all;
clc;
V = 10;
q = 0.25;
k = 9.4*exp(-2500/(8.31*293));

%% RLC circuit
clear
close all;
clc;
R = 1;
L = 2;
C = 1;

%% Mass-spring-damper
clear
close all;
clc;
m = 250;
k = 50;
b = 50;
omega = 1;
A = 0.01;
