%% TAREA 2: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez y Francisca Villegas
% jcbermudez@uc.cl; favillegas@uc.cl

clc; 
clear;
close all;

%% 1. PREAMBULO

% Importar datos y conservar solo con el año 2008 utilizado en la tabla 1
data   = readtable('russian_billionaires.csv');
year08 = 2008;
datos1 = data(data.year == year08, :);

% Removiendo elementos NaN (que tampoco forman parte en el paper)
datos = rmmissing(datos1, 'DataVariables', {'lngdppc', 'lnpop', 'gattwto08'});
clear data datos1;

% Matricez X, Y incluyendo solo los datos que utilizan en la tabla 1 del paper
X = [ones(size(datos,1),1), datos.lngdppc, datos.lnpop, datos.gattwto08];
Y = [datos.numbil0];


%% 2. ESTIMACIÓN DEL MODELO MEDIANTE ML USANDO NEWTON-RAPSON

% El primer valor inicial es mediante OLS, los demás arbitrarios
lnY      = log(1 + Y);
beta_ols = (X'*X)^(-1)*(X'*lnY);

% Prealocación: matriz de 4 valores iniciales distintos
beta0      = NaN(size(X,2),4);
beta0(:,1) = beta_ols(:,1);
beta0(:,2) = beta_ols(:,1) - 0.05;
beta0(:,3) = beta_ols(:,1) + 0.05;
beta0(:,4) = zeros(size(beta0,1),1);

% Preámbulo para la iteración
beta_hat = NaN(size(X,2),4);
converge = NaN(size(beta0,1),1);
iter     = NaN(size(beta0,1),1);
N        = size(Y,1);
error    = 10^-5;

% Desarrollo del Método Newton-Rapson
for j = 1:size(beta0,1)
    
    beta_hat(:,j) = beta0(:,j);
    i = 0;

    tic
    while 1
        % Jacobiano
        aux_J = NaN(size(beta_hat,1), 1, N);
        for k = 1:N
            aux_J(:,:,k) = jacobiano(beta_hat(:,j)', X(k,:), Y(k,1));
        end
        score = sum(aux_J,3);

        % Hessiano
        aux_H = NaN(size(beta_hat,1), size(beta_hat,1), N);
        for k = 1:N
            aux_H(:,:,k) = hessiano(beta_hat(:,j)', X(k,:));
        end
        H = sum(aux_H,3)^(-1);

        beta_hat(:,j) = beta_hat(:,j) - (H*score);
        dif_beta      = abs(H*score);
        
        if dif_beta < error
            break 
        end
        i = i + 1;
    end

    converge(j,1) = toc;
    iter(j,1)     = i;
    disp(['Convergencia alcanzada en ', num2str(converge(j,1)), ' segundos y ', num2str(iter(j,1)), ' iteraciones.']);
end


%% 3. ESTIMACIÓN DE INTERVALOS DE CONFIANZA




















