function Psi = Psi(z)
    zeta = 0.11;%Establish Tuning Parameter
    Psi = (1-exp(-zeta*z))/zeta;
end