
close all
N = 10000;
u = unifrnd(0,1,[N,1])
x = -log((1-u)./u);
sum_X = sum(x);
histogram(x, 100);
