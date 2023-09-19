function validado = validador_aleatorio(x)

    if x <= 0
        validado = 0;
    elseif x >= 1
        validado = 1;
    else
        validado = x;
    end

end