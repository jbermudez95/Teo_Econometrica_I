%% TAREA 1: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez 
% jcbermudez@uc.cl

clc; clear all;

%% INCISO 1: Distribución exacta y Simulación Montecarlo

n = 100;
x_uniforme = rand(n,1);

cdf_exacta = NaN(n,1);

for i = 1:n
    cdf_exacta(i,1) = cdf_exacta_uniforme(x_uniforme(i,1));
end


%% DEFINICIÓN DE FUNCIONES

function cdf_uniforme = cdf_exacta_uniforme(x)
    if x < 0
        cdf_uniforme = 0;
    elseif x > 1
        cdf_uniforme = 1;
    else
        cdf_uniforme = x;
    end
end

