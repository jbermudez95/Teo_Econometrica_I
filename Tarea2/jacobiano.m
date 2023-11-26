function [jacobian] = jacobiano(beta,x,y)
% hessiano.m: Esta función recibe como input el vector de betas y el vector
% de la i-esima fila de la matriz de variables independientes "x_i" para
% evaluar la segunda derivada de la log-likelihood

jacobian = (y-exp(x'*beta))*x;
end