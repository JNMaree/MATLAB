%1D LINEAR FEM SOLVER
%onedlinFEM(12,0,30,10,20,10);
ne=8;
xmin=0;
xmax=50;
k=40;
ti=20;
qi=10;
%function [] = onedlinFEM(ne,xmin,xmax,k,ti,qi)
%SOLVE PDE:
%
% -d/dx[Af(x)*(du/dx)] + Bf(x)*u = Cf(x)
%
%INPUTS:
% ne    :: Number of Elements
% xm    :: Domain Length
% ki    :: Thermal Conduction Coefficient
% ti    :: Temperature Boundary Condition
% qi    :: Thermal Load Flux

%Parameters:                        
nn = ne + 1;                        % Number of Nodes
xx = linspace(xmin,xmax,nn);        % X-coordinates/
%DE functions:
Af = '1';
Bf = '0';
Cf = '0';
%Gaussian Quadrature Defintion:
%3 Point
GQp = [-sqrt(3/5),0,sqrt(3/5)];
GQw = [5/9,8/9,5/9];
GQn = length(GQp);

%Shape functions and Gradients @ GQ points
S = zeros(2,GQn);
dS = zeros(2,GQn);

for e=1:GQn
   S(1,e) = (1-GQp(e))/2;
   S(2,e) = (1+GQp(e))/2;
   dS(1,e) = -0.5;
   dS(2,e) = +0.5;
end

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
        xE = GQp(q)*((x2-x1)/2) + (x2+x1)/2;
        Ac = eval(Af)*k;
        Bc = eval(Bf);
        Cc = eval(Cf);
        for i=1:2
            for j=1:2
                Ke(i,j) = Ke(i,j) + ...
                         (Ac*(dS(i,q)/jac*dS(j,q)/jac)+...
                          Bc*(S(j,q)*S(i,q)))*jac*GQw(q);
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
else
    
end
%RIGHT boundary
if rtype == 1
    F(nn) = rt1;
    K(nn,:) = 0;
    K(nn,nn) = 1;
elseif rtype == 2
    F(nn) = F(nn) + rq1;
else
    
end

%Solve the Global System
uT = K\F;

%POST: Plot FE Solution
plot(xx,uT);
grid on;
title('FE Solution of model PDE');
xlabel('x');
ylabel('T');
%end
