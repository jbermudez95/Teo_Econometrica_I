%% TAREA 1: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez 
% jcbermudez@uc.cl

clc; 
clear;
close all;



%% INCISO 1: Distribución exacta y Simulación Montecarlo

n = 1e3;

dominio_generado = linspace(-2,2,n);
valor = cdf_exacta(dominio_generado, n);


figure(1)
plot(dominio_generado, valor)
axis([-3 3 -0.5 1.5])

% Simulaciones %

n = [1e3, 1e5, 1e7];
arreglo_0_1 = linspace(0,1,50);
valores_cdf = cdf_exacta(arreglo_0_1, 50);


sim1 = rand(n(1),1);


figure(2)
histogram(sim1,'Normalization','cdf', NumBins = 50)
hold on
plot(arreglo_0_1, valores_cdf)
legend()


sim2 = rand(n(2),1);

figure(3)
histogram(sim2,'Normalization','cdf', NumBins = 50)
hold on
plot(arreglo_0_1, valores_cdf)
legend()


sim3 = rand(n(3),1);

figure(4)
histogram(sim3,'Normalization','cdf', NumBins = 50)
hold on
plot(arreglo_0_1, valores_cdf)
legend()


%histogram(B10_5,'Normalization','cdf', NumBins = 100)
%% INCISO 2: Distribución exacta y Simulación Montecarlo

%Parte 1
N = [1, 10, 100, 1000];
S = 10000;

% Inicializar matrices para almacenar resultados
means_unif = zeros(length(N), S);
means_exp = zeros(length(N), S);
means_bern = zeros(length(N) - 1, S);  % Excluir N=1 para Bernoulli
var_unif = zeros(length(N) - 1, S);   % Excluir N=1 para la varianza
var_exp = zeros(length(N) - 1, S);    % Excluir N=1 para la varianza
var_bern = zeros(length(N) - 1, S);   % Excluir N=1 para la varianza

for i = 1:length(N)
    % Generar muestras para cada distribución y tamaño de muestra
    unif_samples = rand(N(i), S);
    exp_samples = exprnd(3, N(i), S);
    if i > 1
        bern_samples = binornd(1, 0.7, N(i), S);
    end
    
    % Calcular la media muestral y la varianza muestral
    means_unif(i, :) = mean(unif_samples);
    means_exp(i, :) = mean(exp_samples);
    if i > 1
        means_bern(i-1, :) = mean(bern_samples);
        var_unif(i-1, :) = var(unif_samples);
        var_exp(i-1, :) = var(exp_samples);
        var_bern(i-1, :) = var(bern_samples);
    end
end

% Parte 3
figure;

% Gráfico de la media muestral para uniforme
subplot(3, 2, 1);
histogram(means_unif(1, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Uniforme (n=1)');

subplot(3, 2, 3);
histogram(means_unif(2, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Uniforme (n=10)');

subplot(3, 2, 5);
histogram(means_unif(3, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Uniforme (n=100)');

subplot(3, 2, 2);
histogram(means_exp(1, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Exponencial (n=1)');

subplot(3, 2, 4);
histogram(means_exp(2, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Exponencial (n=10)');

subplot(3, 2, 6);
histogram(means_exp(3, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Exponencial (n=100)');

% Parte 6
figure;

subplot(2, 2, 1);
histogram(zeros(S), 'Normalization', 'pdf');
axis([0 0.2 0 60]);
title('Distribución de la varianza muestral para Uniforme (n=1)');

subplot(2, 2, 2);
histogram(var_unif(1, :), 'Normalization', 'pdf');
axis([0 0.2 0 60]);
title('Distribución de la varianza muestral para Uniforme (n=10)');

subplot(2, 2, 3);
histogram(var_unif(2, :), 'Normalization', 'pdf');
axis([0 0.2 0 60]);
title('Distribución de la varianza muestral para Uniforme (n=100)');

subplot(2, 2, 4);
histogram(var_unif(3, :), 'Normalization', 'pdf');
axis([0 0.2 0 60]);
title('Distribución de la varianza muestral para Uniforme (n=1000)');

%% %% INCISO 2: Distribución asintótica de la media y varianza muestral.


clc; clear all;

% Parte 1
N = [1, 10, 100, 1000];
S = 10000;

% Inicializar matrices para almacenar resultados
means_unif = zeros(length(N), S);
means_exp = zeros(length(N), S);
means_bern = zeros(length(N) - 1, S);  % Excluir N=1 para Bernoulli
var_unif = zeros(length(N) - 1, S);   % Excluir N=1 para la varianza
var_exp = zeros(length(N) - 1, S);    % Excluir N=1 para la varianza
var_bern = zeros(length(N) - 1, S);   % Excluir N=1 para la varianza

for i = 1:length(N)
    % Generar muestras para cada distribución y tamaño de muestra
    unif_samples = rand(N(i), S);
    exp_samples = exprnd(3, N(i), S);
    if i > 1
        bern_samples = binornd(1, 0.7, N(i), S);
    end
    
    % Calcular la media muestral y la varianza muestral
    means_unif(i, :) = mean(unif_samples);
    means_exp(i, :) = mean(exp_samples);
    if i > 1
        means_bern(i-1, :) = mean(bern_samples);
        var_unif(i-1, :) = var(unif_samples);
        var_exp(i-1, :) = var(exp_samples);
        var_bern(i-1, :) = var(bern_samples);
    end
end

% Parte 3
figure;

% Gráfico de la media muestral para uniforme
subplot(3, 2, 1);
histogram(means_unif(1, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Uniforme (n=1)');

subplot(3, 2, 3);
histogram(means_unif(2, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Uniforme (n=10)');

subplot(3, 2, 5);
histogram(means_unif(3, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Uniforme (n=100)');

subplot(3, 2, 2);
histogram(means_exp(1, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Exponencial (n=1)');

subplot(3, 2, 4);
histogram(means_exp(2, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Exponencial (n=10)');

subplot(3, 2, 6);
histogram(means_exp(3, :), 'Normalization', 'pdf');
title('Distribución de la media muestral para Exponencial (n=100)');

% Parte 6
figure;

subplot(2, 2, 1);
histogram(zeros(S), 'Normalization', 'pdf');
axis([0 0.2 0 2]);
title('Distribución de la varianza muestral para Uniforme (n=1)');

subplot(2, 2, 2);
histogram(var_unif(1, :), 'Normalization', 'pdf');
axis([0 0.2 0 20]);
title('Distribución de la varianza muestral para Uniforme (n=10)');

subplot(2, 2, 3);
histogram(var_unif(2, :), 'Normalization', 'pdf');
axis([0 0.2 0 60]);
title('Distribución de la varianza muestral para Uniforme (n=100)');

subplot(2, 2, 4);
histogram(var_unif(3, :), 'Normalization', 'pdf');
axis([0 0.2 0 160]);
title('Distribución de la varianza muestral para Uniforme (n=1000)');


%% 







