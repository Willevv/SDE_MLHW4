%--------config----------------
clear all
close all
rand('seed', 1)
n = 1000; %Number of iterations. 
theta0 =1;
dt = 0.04;
THETA = [];
F = 1:1001;
THETA(1) = 1;

for i = 1:n
    y = randn();
    theta_upd = THETA(i) - dt*df(THETA(i), y);
    THETA = [THETA theta_upd];
end

plot(F, THETA)
title('Plot of Theta_n for n = 1,2, ..., N, dt = 0.04')
xlabel('Number of iterations')
ylabel('Theta_n')

function df = df(theta,y)
    df = 2*(theta-y);
end

function f = f(theta, y)
    f = abs(theta-y).^2; 
end
