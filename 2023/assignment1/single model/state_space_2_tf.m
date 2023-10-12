%% state-space to transfer function
m = 2;
b = 3;
k = 1;

A = [ 0 1; -k/m -b/m];
B = [ 0 1/m]';
C = [1 0];
D = 0;

sys = ss(A,B,C,D);
[num,den] = ss2tf(A,B,C,D,1);

H = tf(num*2,den*2)


%% bode plot

bode(H)

%%