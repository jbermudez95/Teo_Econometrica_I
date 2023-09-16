%% TAREA 1: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez 
% jcbermudez@uc.cl

clc; clear all;

%% INCISO 1: Distribución exacta y Simulación Montecarlo

%n = 100;
%x_uniforme = rand(n,1);

%cdf_exacta = NaN(n,1);

%for i = 1:n
    cdf_exacta(i,1) = cdf_exacta_uniforme(x_uniforme(i,1));
%end


%%% DEFINICIÓN DE FUNCIONES

%function cdf_uniforme = cdf_exacta_uniforme(x)
    %if x < 0
       % cdf_uniforme = 0;
    %elseif x > 1
        %cdf_uniforme = 1;
    %else
        %cdf_uniforme = x;
    %end
%end

clc; clear all;

% 1. Generar variable aleatoria con distribución U[0,1]
n = 100;
x = rand(n, 1);

% 2. Graficar la Distribución Exacta
x_values = linspace(0, 1, 1000); % Valores de x entre 0 y 1
cdf_values = cdf_exacta(x_values); % Calcula la CDF exacta

figure;

subplot(1, 3, 1);
plot(x_values, cdf_values, 'b');
title('Distribución Exacta');
xlabel('x');
ylabel('CDF');

% 3. Simulación Montecarlo
num_simulaciones = [1000, 100000, 10000000];

for i = 1:length(num_simulaciones)
    N = num_simulaciones(i);
    samples = rand(N, 1); % Generar N muestras aleatorias de U[0,1]
    ecdf = cumsum(samples) / N; % CDF empírica

    % 4. Graficar la Distribución Montecarlo
    subplot(1, 3, i + 1);
    plot(x_values, cdf_values, 'b', x_values, ecdf, 'r');
    title(['Montecarlo (', num2str(N), ' simulaciones)']);
    xlabel('x');
    ylabel('CDF');
    legend('Exacta', 'Montecarlo', 'Location', 'northwest');
end

% 5. Función de Distribución Acumulativa (CDF) Exacta
function cdf = cdf_exacta(x)
    cdf = x .* (x >= 0 & x <= 1);
end
%% % Parte 1
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
title('Distribución de la varianza muestral para Uniforme (n=1000)');}

%% 







