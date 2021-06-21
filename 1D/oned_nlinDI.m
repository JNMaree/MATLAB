%1D NONLINEAR FEM SOLVER
%DIRECT ITERATION
%input parameters:
ne=8;
xmin=0;
xmax=50;

k=40;

ti=20;
qi=10;

%nonlinear paramaters
itrMax = 20;
eTol = 1e-3;
%Thermal Conductivity as a Function of Temperature
%K(T) = kM*T + kC
kM = 2;
kC = 1;

%SOLVE PDE:
%
% -d/dx[Af(x)*(du/dx)] = Cf(x)
%
%INPUTS:
% ne    :: Number of Elements
% xmin  :: Starting X position
% xmax  :: Ending X position
% k     :: Thermal Conduction Coefficient
% ti    :: Temperature Boundary Condition
% qi    :: Thermal Load Flux
%
%NL INPUTS:
% itrMax    :: Max Iterations Permitted
% eTol      :: Convergence Error Tolerance
% 

%Parameters:                        
nn = ne + 1;                        % Number of Nodes
xx = linspace(xmin,xmax,nn);        % X-coordinates/
%Iteration Counter Definition
itr = 0;
loopr = true;
%DE functions:
Af = '1';
Cf = '0';
%BC definitions:
ltype = 1;  %1: Dirichlet Boundary Condition  
rtype = 2;  %2: Neumann Boundary Condition
            %3: Mixed Boundary Condition
lt1 = ti;
rt1 = 0;
lq1 = 0;
rq1 = qi;

%Global matrix Definitions:
K = sparse(nn,nn);
F = sparse(nn,1);

KZ = K;
FZ = F;

%Gaussian Quadrature Defintion:
%3 Point
GQp = [-sqrt(3/5),0,sqrt(3/5)];
GQw = [5/9,8/9,5/9];
GQn = length(GQp);

%Shape functions and Gradients @ GQ points
S = zeros(2,GQn);
dS = zeros(2,GQn);
%Shape Function Definiton:
for e=1:GQn
   S(1,e) = (1-GQp(e))/2;
   S(2,e) = (1+GQp(e))/2;
   dS(1,e) = -0.5;
   dS(2,e) = +0.5;
end

%INITIALISATION
for e=1:ne
    x1 = xx(e);
    x2 = xx(e+1);
    %element jacobian
    jac = (x2-x1)/2;
    
    %local matrices
    Ke = zeros(2,2);
    Fe = zeros(2,1);
    %loop through GQ points
    for q=1:GQn
        %Evaluate Current X position
        xE = GQp(q)*((x2-x1)/2) + (x2+x1)/2;
        %First Iteration Conditions (LINEAR)
        Ac = eval(Af)*k;
        Cc = eval(Cf);
        
        for i=1:2
            for j=1:2
                Ke(i,j) = Ke(i,j) + ...
                          Ac*(dS(i,q)/jac*dS(j,q)/jac)...
                          *jac*GQw(q);
            end
            Fe(i) = Fe(i) + Cc*S(i,q)*jac*GQw(q);
        end
    end
    
    %Assemble Local to Global
    KZ(e,e)      = KZ(e,e) + Ke(1,1);
    KZ(e+1,e)    = KZ(e+1,e) + Ke(2,1);
    KZ(e,e+1)    = KZ(e,e+1) + Ke(1,2);
    KZ(e+1,e+1)  = KZ(e+1,e+1) + Ke(2,2);
    
    FZ(e)        = FZ(e) + Fe(1);
    FZ(e+1)      = FZ(e+1) + Fe(2);
end

%LEFT boundary
if ltype == 1
    FZ(1) = lt1;
    KZ(1,:) = 0;
    KZ(1,1) = 1;
elseif ltype == 2
    FZ(1) = FZ(1) + lq1;
end
%RIGHT boundary
if rtype == 1
    FZ(nn) = rt1;
    KZ(nn,:) = 0;
    KZ(nn,nn) = 1;
elseif rtype == 2
    FZ(nn) = FZ(nn) + rq1;
end

uR = KZ\FZ;

%NONLINEAR UTERATION LOOP
while loopr
    %Increment Iteration Counter
    itr = itr + 1;
    %Local Element Matrix Definitions:
    for e=1:ne
        x1 = xx(e);
        x2 = xx(e+1);
        %element jacobian
        jac = (x2-x1)/2;
        
        %local matrices
        Ke = zeros(2,2);
        Fe = zeros(2,1);
        %loop through GQ points
        for q=1:GQn
            %Evaluate Current X position
            xE = GQp(q)*((x2-x1)/2) + (x2+x1)/2;
            
            Ac = eval(Af)*(kM*(uR(e)) + kC);
            Cc = eval(Cf);
            
            for i=1:2
                for j=1:2
                    Ke(i,j) = Ke(i,j) + ...
                              Ac*(dS(i,q)/jac*dS(j,q)/jac)...
                              *jac*GQw(q);
                end
                Fe(i) = Fe(i) + Cc*S(i,q)*jac*GQw(q);
            end
        end
        
        %Assemble Local to Global
        K(e,e)      = K(e,e) + Ke(1,1);
        K(e+1,e)    = K(e+1,e) + Ke(2,1);
        K(e,e+1)    = K(e,e+1) + Ke(1,2);
        K(e+1,e+1)  = K(e+1,e+1) + Ke(2,2);
        
        F(e)        = F(e) + Fe(1);
        F(e+1)      = F(e+1) + Fe(2);
    end
    
    %BC Applications:
    %LEFT boundary
    if ltype == 1
        F(1) = lt1;
        K(1,:) = 0;
        K(1,1) = 1;
    elseif ltype == 2
        F(1) = F(1) + lq1;
    end
    %RIGHT boundary
    if rtype == 1
        F(nn) = rt1;
        K(nn,:) = 0;
        K(nn,nn) = 1;
    elseif rtype == 2
        F(nn) = F(nn) + rq1;
    end
    
    %Solve the Global System
    uR = KZ\FZ;
    
    %Residual Calculation
    R = K*uR - F;
    KZ = K;
    FZ = F;
    resi = max(R(:));
    
    %Exit Nonlin Loop Condition
    if resi < eTol
        loopr = false;
        disp('Convergence Reached');
    elseif itr >= itrMax
        loopr = false;
        disp('MAX Iterations Reached');
        disp('Convergence NOT Reached');
    end
end

%POST: Plot FE Solution
plot(xx,uR,'-o');
grid on;
title('FE Solution of model PDE');
xlabel('x');
ylabel('T');
%end
