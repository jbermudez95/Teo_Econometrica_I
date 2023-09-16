%% TAREA 1: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez 
% jcbermudez@uc.cl

clc; 
clear;
close all;

rng('default')  % Para reproducibilidad de los números aleatorios

%% INCISO 1: Distribución exacta y Simulación Montecarlo

n = 100;
x_uniforme = rand(n,1);

cdf_exacta = NaN(n,1);

for i = 1:n
    cdf_exacta(i,1) = cdf_exacta_uniforme(x_uniforme(i,1));
end

%% INCISO 3: DISTRIBUCIÓN DE KERNEL

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
histogram(x_exp_100,'Normalization','pdf','FaceAlpha',0.2,'BinWidth',1)

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
histogram(x_exp_1000,'Normalization','pdf','FaceAlpha',0.2,'BinWidth',1)

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

