function Psi_dx = Psi_dx(z)
    zeta = 0.11; %Establish tuning parameter
    Psi_dx = exp(-zeta*z);
end