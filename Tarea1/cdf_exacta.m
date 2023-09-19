function valores_prob = cdf_exacta(X, n)
    valores_prob = zeros(1,n);
    for indice = 1:n
        paso = X(indice);
        intermedio = validador_aleatorio(paso);
        valores_prob(indice) = intermedio;
    end
end