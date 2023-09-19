function [betas] = mco(x,y)
    betas = (x'*x)^(-1)*(x'*y);
end