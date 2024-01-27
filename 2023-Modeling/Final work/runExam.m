clear
load modeldata.mat
prompt = 'Please Enter the numeric part of your student number!:  ';
studentNo = input(prompt);
k = find(modeldata.studentList == studentNo);
u1 = modeldata.u{k,1}; 
y1 = modeldata.y{k,1};
u2 = modeldata.u{k,2}; 
y2 = modeldata.y{k,2};
u3 = modeldata.u{k,3}; 
y3 = modeldata.y{k,3};
clear modeldata prompt studentNo k
data1 = iddata(y1,u1,1);
nk1 = delayest(data1);
disp(nk1);
data2 = iddata(y2,u2,1);
nk2 = delayest(data2);
disp(nk2);
data3 = iddata(y3,u3,1);
nk3 = delayest(data3);
disp(nk3);