function [jacobian] = jacobiano(beta,x,y)
% jacobiano.m: Esta función recibe como input el vector de betas, el vector
% de la i-esima fila de la matriz de variables independientes "x_i", y el 
% i-ésimo elemento de la variable dependiente "y_i". Evalua la CPO de la 
% log-likelihood y entrega el Jacobiano.

jacobian = y*x'-exp(beta*x')*x';
end