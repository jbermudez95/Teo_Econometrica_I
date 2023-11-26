%% TAREA 2: TEORÍA ECONOMÉTRICA I
% Jose Carlo Bermúdez: jcbermudez@uc.cl 
% Francisca Villegas: favillegas@uc.cl

clc; 
clear;
close all;

% Detalles para los gráficos
tx  = {'Interpreter','Latex','FontSize', 14};
tx1 = {'Interpreter','Latex','FontSize', 12};

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


%% 3. INTERVALOS DE CONFIANZA: BOOTSTRAP NO PARAMÉTRICO

% Prealocación 
N_simul        = 1000;
beta_aux       = NaN(size(beta0,1), N_simul);
beta_aux(:,1)  = beta_ols(:,1);
beta_bootstrap = NaN(size(beta0,1), N_simul);
data_bootstrap = [datos.numbil0, ones(size(datos,1),1), datos.lngdppc, datos.lnpop, datos.gattwto08];

% Bootstrapping
for j = 1:N_simul

    % Remuestreo con remplazo 
    muestra_b = datasample(data_bootstrap, N);
    Y_b       = muestra_b(:,1);
    X_b       = muestra_b(:,2:end);

    i = 1;
    while 1
        % Jacobiano
        aux_J_b = NaN(size(beta_aux,1), 1, N);
        for k = 1:N
            aux_J_b(:,:,k) = jacobiano(beta_aux(:,i)', X_b(k,:), Y_b(k,1));
        end
        score_b = sum(aux_J_b,3);

        % Hessiano
        aux_H_b = NaN(size(beta_aux,1), size(beta_aux,1), N);
        for k = 1:N
            aux_H_b(:,:,k) = hessiano(beta_aux(:,i)', X_b(k,:));
        end
        H_b = sum(aux_H_b,3)^(-1);

        beta_aux(:,i+1) = beta_aux(:,i) - (H_b*score_b);
        dif_beta_b      = abs(H_b*score_b);
        
        if dif_beta_b < error
            break 
        end
        i = i + 1;
    end

    beta_bootstrap(:,j) = beta_aux(:,i);
end

% Calculamos los IC mediante el método de Hall
beta_test = beta_bootstrap - repmat(beta_hat(:,1), 1, N_simul);


% Distribución para Beta 0
beta0_bootstrap = sort(beta_test(1,:));
beta0_low       = beta_hat(1,1) - beta0_bootstrap(975);
beta0_up        = beta_hat(1,1) - beta0_bootstrap(25);

figure(1)
histogram(beta_bootstrap(1, :),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Boostrapping para $\hat{\beta}_0$',tx{:})
xline(beta_hat(1,1),'--r','$\hat{\beta}_0$', tx1{:},'LabelOrientation','horizontal')
xline(beta0_low,'--b','$\hat{\beta}_0^{1-\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
xline(beta0_up,'--b','$\hat{\beta}_0^{\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
legend('off')
ylabel('Distribucion',tx1{:})
exportgraphics(figure(1),'beta0_bootstrap.pdf')


% Distribución para Beta 1
beta1_bootstrap = sort(beta_test(2,:));
beta1_low       = beta_hat(2,1) - beta1_bootstrap(975);
beta1_up        = beta_hat(2,1) - beta1_bootstrap(25);

figure(2)
histogram(beta_bootstrap(2, :),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Boostrapping para $\hat{\beta}_1$',tx{:})
xline(beta_hat(2,1),'--r','$\hat{\beta}_1$', tx1{:},'LabelOrientation','horizontal')
xline(beta1_low,'--b','$\hat{\beta}_1^{1-\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
xline(beta1_up,'--b','$\hat{\beta}_1^{\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
legend('off')
ylabel('Distribucion',tx1{:})
exportgraphics(figure(2),'beta1_bootstrap.pdf')


% Distribución para Beta 2
beta2_bootstrap = sort(beta_test(3,:));
beta2_low       = beta_hat(3,1) - beta2_bootstrap(975);
beta2_up        = beta_hat(3,1) - beta2_bootstrap(25);

figure(3)
histogram(beta_bootstrap(3, :),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Boostrapping para $\hat{\beta}_2$',tx{:})
xline(beta_hat(3,1),'--r','$\hat{\beta}_2$', tx1{:},'LabelOrientation','horizontal')
xline(beta2_low,'--b','$\hat{\beta}_2^{1-\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
xline(beta2_up,'--b','$\hat{\beta}_2^{\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
legend('off')
ylabel('Distribucion',tx1{:})
exportgraphics(figure(3),'beta2_bootstrap.pdf')


% Distribución para Beta 3
beta3_bootstrap = sort(beta_test(4,:));
beta3_low       = beta_hat(4,1) - beta3_bootstrap(975);
beta3_up        = beta_hat(4,1) - beta3_bootstrap(25);

figure(4)
histogram(beta_bootstrap(4, :),'FaceColor','b','EdgeColor','none','FaceAlpha',0.2)
title('Boostrapping para $\hat{\beta}_3$',tx{:})
xline(beta_hat(4,1),'--r','$\hat{\beta}_3$', tx1{:},'LabelOrientation','horizontal')
xline(beta3_low,'--b','$\hat{\beta}_3^{1-\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
xline(beta3_up,'--b','$\hat{\beta}_3^{\frac{\alpha}{2}}$', tx1{:},'LabelOrientation','horizontal')
legend('off')
ylabel('Distribucion',tx1{:})
exportgraphics(figure(4),'beta3_bootstrap.pdf')

