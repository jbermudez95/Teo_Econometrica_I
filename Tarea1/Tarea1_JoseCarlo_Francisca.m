%% TAREA 1: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez y Francisca Villegas
% jcbermudez@uc.cl; favillegas@uc.cl

clc; 
clear;
close all;

%% EJERCICIO 1: DISTRIBUCIÓN EXACTA Y SIMULACIÓN MONTECARLO

rng('default') % Para reproducibilidad de datos aleatorios
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
legend('Simulado', 'CDF exacta','Location','northwest')

sim2 = rand(n(2),1);

figure(3)
histogram(sim2,'Normalization','cdf', NumBins = 50)
hold on
plot(arreglo_0_1, valores_cdf)
legend('Simulado', 'CDF exacta','Location','northwest')


sim3 = rand(n(3),1);

figure(4)
histogram(sim3,'Normalization','cdf', NumBins = 50)
hold on
plot(arreglo_0_1, valores_cdf)
legend('Simulado', 'CDF exacta','Location','northwest')


%% EJERCICIO 2: DISTRIBUCIÓN ASINTÓTICA DE LA MEDIA Y VARIANZA MUESTRAL

clc; clear all;
rng('default') % Para reproducibilidad de datos aleatorios

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


%% EJERCICIO 3: DISTRIBUCIÓN DE KERNEL

clc; clear; close all;
rng('default')

% Simulación para un VA con distribución exponencial (λ=3)
lambda = 3;
x_exp_100  = exprnd(lambda,100,1);
x_exp_1000 = exprnd(lambda,1000,1);

smooth = {'normal','box','triangle','epanechnikov'};

for i = 1:length(smooth)
    [pdf100{i},x100{i}]   = ksdensity(x_exp_100,'Kernel',smooth{i});
    [pdf1000{i},x1000{i}] = ksdensity(x_exp_1000,'Kernel',smooth{i});
end

% Gráficas
lstyle = {'-','-.','--','-..'};
lcolor = {'c','g','r','b'};
tx  = {'Interpreter','Latex','FontSize', 10};
tx1 = {'Interpreter','Latex','FontSize', 7};

figure(3)
histogram(x_exp_100,'Normalization','pdf','EdgeColor','none','FaceAlpha',0.2,'BinWidth',1)

hold on
for i = 1:length(smooth)
    plot(x100{1,i},pdf100{1,i},lstyle{i},'LineWidth',2, 'Color',lcolor{i})
end
hold off

legend('Densidad Real','Kernel Normal','Kernel Box','Kernel Triangle','Kernel Epanechnikov','Location','northeast', tx1{:});
legend('boxoff')
xlabel('$x_{i}$',tx1{:})
ylabel('PDF $(x_{i})$',tx1{:})
sgtitle('Simulacion para $n=100$',tx{:})
exportgraphics(figure(3),'inciso_c100.pdf')

figure(4)
histogram(x_exp_1000,'Normalization','pdf','EdgeColor','none','FaceAlpha',0.2,'BinWidth',1)

hold on
for i = 1:length(smooth)
    plot(x1000{1,i},pdf1000{1,i},lstyle{i},'LineWidth',2, 'Color',lcolor{i})
end
hold off

legend('Densidad Real','Kernel Normal','Kernel Box','Kernel Triangle','Kernel Epanechnikov','Location','northeast', tx1{:});
legend('boxoff')
xlabel('$x_{i}$',tx1{:})
ylabel('PDF $(x_{i})$',tx1{:})
sgtitle('Simulacion para $n=1,000$',tx{:})
exportgraphics(figure(4),'inciso_c1000.pdf')

%% EJERCICIO 4: ESTIMACIÓN POR MCO

clc; clear; close all;
rng('default')

% ========== 4.1. Algoritmo de estimación mediante Montecarlo ========== 

% Simulando el primer beta
N       = 10^3;    
mu_x    = 5;
sigma_x = 1.5;
x       = mu_x*ones(N,1) + sigma_x*randn(N,1);                             % Simulación de la variable independiente

sigma_ep = 2;                                                              % Simulación de los errores con media cero y varianza arbitraria
epsilon  = normrnd(0, sigma_ep, N, 1);

X    = [ones(N,1),x];
beta = [1.5,2.8]';
Y    = X*beta + epsilon;                                                   % Simulación de la variable dependiente

betas_hat = mco(X,Y);

% Simulaciones Montecarlo (MC) para beta
N_MC     = 10^4;
betas_MC = NaN(N_MC,2);

epsilon_MC = sigma_ep*randn(N, N_MC);
Y_MC = repmat(X*beta,1,N_MC) + epsilon_MC;

for i = 1:N_MC
    betas_MC(i,:) = ((X'*X)^(-1)*(X'*Y_MC(:,i)))';
end

% Comparativa gráfica para las estimaciones de beta
tx  = {'Interpreter','Latex','FontSize', 10};
tx1 = {'Interpreter','Latex','FontSize', 8};

figure(5)
histogram(betas_MC(:, 1),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Estimaciones para $\hat{\beta}_0$',tx{:})
xline(mean(betas_MC(:,1)),'--r',LineWidth= 2)
xline(betas_hat(1,1),'--b')
xline(beta(1,1))
legend('$\hat{\beta}_0$ Montecarlo','Media $\hat{\beta}_0$ Montecarlo', ...
    '$\hat{\beta}_0$ MCO','True $\beta_0$','Location','northwest', tx1{:});
legend('boxoff')
xlabel('$\hat{\beta}_0$',tx1{:})
ylabel('Distribucion',tx1{:})
exportgraphics(figure(5),'beta0_montecarlo.pdf')

figure(6)
histogram(betas_MC(:,2),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Estimaciones para $\hat{\beta}_1$',tx{:})
xline(mean(betas_MC(:,2)),'--r',LineWidth=2)
xline(betas_hat(2,1),'--b')
xline(beta(2,1))
legend('$\hat{\beta}_1$ Montecarlo','Media $\hat{\beta}_1$ Montecarlo', ...
    '$\hat{\beta}_1$ MCO','True $\beta_1$','Location','northwest', tx1{:});
legend('boxoff')
xlabel('$\hat{\beta}_1$',tx1{:})
ylabel('Distribucion',tx1{:})
exportgraphics(figure(6),'beta1_montecarlo.pdf')

% Simulaciones Montecarlo (MC) para la varianza
varianza     = var(epsilon)*(diag((X'*X)^(-1)))';
varianza_hat = var(betas_MC);
sq_error_MC  = NaN(size(Y_MC, 1), 2);
betas_MC     = betas_MC';

for j = 1:N_MC
  sq_error_MC(:, j) = (Y_MC(:,j) - X*(betas_MC(:,j))).^2;      
end 

n = size(sq_error_MC, 1);
k = size(X, 2);
varianza_errores_MC = (sum(sq_error_MC)./(n-k))';

varianza_beta_MC = NaN(N_MC, 2);

for i = 1:N_MC
    varianza_beta_MC(i, :) = varianza_errores_MC(i).*(diag((X'*X)^(-1)))';
end

% Comparativa gráfica para las estimaciones de la varianza
figure(7)
histogram(varianza_beta_MC(:, 1),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Estimaciones para $\hat{\sigma}^{2}$ de $\hat{\beta}_0$',tx{:})
xline(mean(varianza_beta_MC(:,1)),'--r',LineWidth= 2)
xline(varianza_hat(1,1),'--b')
xline(varianza(1,1))
legend('$\hat{\beta}_0$ Montecarlo','Media $\hat{\sigma}^{2},\hat{\beta}_0$ Montecarlo', ...
    '$\hat{\sigma}^{2}$ MCO','True $\hat{\sigma}^{2}$','Location','northwest', tx1{:});
legend('boxoff')
xlabel('$\hat{\sigma}^{2}, \hat{\beta}_0$',tx1{:})
ylabel('Distribucion',tx1{:})
exportgraphics(figure(7),'beta0_var_montecarlo.pdf')

figure(8)
histogram(varianza_beta_MC(:, 2),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Estimaciones para $\hat{\sigma}^{2}$ de $\hat{\beta}_1$',tx{:})
xline(mean(varianza_beta_MC(:,2)),'--r',LineWidth= 2)
xline(varianza_hat(1,2),'--b')
xline(varianza(1,2))
legend('$\hat{\beta}_1$ Montecarlo','Media $\hat{\sigma}^{2},\hat{\beta}_1$ Montecarlo', ...
    '$\hat{\sigma}^{2}$ MCO','True $\hat{\sigma}^{2}$','Location','northwest', tx1{:});
legend('boxoff')
xlabel('$\hat{\sigma}^{2}, \hat{\beta}_1$',tx1{:})
ylabel('Distribucion',tx1{:})
exportgraphics(figure(8),'beta1_var_montecarlo.pdf')

% ========== 4.2 Contraste de hipótesis para los 10,000 betas estimados ========== 
betas_MC = betas_MC';
beta_verdadero = repmat(beta',N_MC,1);
t_estimado     = NaN(size(beta_verdadero));
contraste      = NaN(size(beta_verdadero));

for j=1:N_MC
    for k=1:size(contraste, 2)
        t_estimado(j, k) = (betas_MC(j,k) - beta_verdadero(j,k))/sqrt(varianza_beta_MC(j,k)); 
        if abs(t_estimado(j,k)) > abs(tinv(0.025, n - k))
            contraste(j,k) = 1;
        else
            contraste(j,k) = 0;
        end
    end
end 

n_rechazos_h0 = (sum(contraste)/N_MC)*100;
for i = 1:length(n_rechazos_h0)
    if i == 1
        disp(['La H0 para Beta 0 se rechaza ', num2str(n_rechazos_h0(i)), '% veces']);
    else
        disp(['La H0 para Beta 1 se rechaza ', num2str(n_rechazos_h0(i)), '% veces']);
    end
end  

% ========== 4.3 Estimación del beta para nuestra muestra de datos ========== 
datos = readtable('etr_bachas_zucman.csv');

log_ndp = log(datos.ndp_usd);
X_datos = [ones(length(log_ndp),1),log_ndp];
Y_datos = datos.ETR_L;

beta_datos = mco(X_datos, Y_datos);

% ========== 4.4 Contraste de hipótesis a los parámetros ========== 
var_errores_datos = sum((Y_datos - X_datos*beta_datos).^2)/(size(X_datos, 1) - size(X_datos, 2));
var_beta_datos    = var_errores_datos*((X_datos'*X_datos)^(-1));

t_datos = beta_datos./diag(var_beta_datos); 
n_beta = [0 1];

for i = 1:length(beta_datos)
    if abs(t_datos(i)) > abs(tinv(0.025, size(X_datos, 1) - size(X_datos, 2)))
        disp(['Beta ', num2str(n_beta(i)), ' resulta estadísticamente significativo.']);
    else
        disp(['Beta ', num2str(n_beta(i)), ' no es estadísticamente significativo.']);
    end
end    

