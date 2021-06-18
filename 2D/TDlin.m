%PROBLEM DEFINITION

a = 1;
f = 0;
%MESH DEFINITIONS
%Node Definitions
x = [0,2,6,7];
y = [0,1,3,4,6];

nx=length(x);
ny=length(y);
nn=nx*ny;
nxy=zeros(nn,2);

nc=1;
for j=1:ny
   for i=1:nx
        nxy(nc,1) = x(i);
        nxy(nc,2) = y(j);
        nc = nc + 1;
   end
end
%
%Element Definitions
ne=16;
enp=zeros(ne,4);

enp(1,:) = [1,2,6,0];
enp(2,:) = [1,6,5,0];
enp(3,:) = [2,3,7,6];
enp(4,:) = [3,4,7,0];
enp(5,:) = [4,8,7,0];
enp(6,:) = [5,6,10,0];
enp(7,:) = [5,10,9,0];
enp(8,:) = [7,8,12,11];
enp(9,:) = [9,10,14,13];
enp(10,:) = [11,12,16,0];
enp(11,:) = [11,16,15,0];
enp(12,:) = [13,14,17,0];
enp(13,:) = [14,18,17,0];
enp(14,:) = [14,15,19,18];
enp(15,:) = [15,20,19,0];
enp(16,:) = [15,16,20,0];
%
%Boundary Condition Definitions
bcn(1,:) = [1,5,9,13,17]; %BC applied Nodes
bcn(2,:) = [30,30,30,30,30];

bce(1,:) = [16,10,8,5];   %BC applied Elements
bce(2,:) = [400,400,400,400];
bce(3,:) = [2,2,2,3];       %faceIDs


%function calls
[Ki,Fi] = calculate(nn,ne,enp,nxy,a,f);
[K,F] = boundCond(bcn,bce,Ki,Fi,enp,nxy);
U = K\F;
postProcessing(ne,nxy,enp,U);


%GAUSSIAN QUADRATURE & SHAPE FUNCTION DEFINITION
function [GQw,S,dS] = defineGQS(nen)
    %4 Point Gaussian Quadrature for both ele types
    if nen==3    %TRI element
       %s-Value             t-Value
        GQp(1,1) = 1/3;     GQp(1,2) = 1/3;
        GQp(2,1) = 0.6;     GQp(2,2) = 0.2;
        GQp(3,1) = 0.2;     GQp(3,2) = 0.6;
        GQp(4,1) = 0.2;     GQp(4,2) = 0.2;
        
        GQw(1) = -27/96;
        GQw(2) = 25/96;
        GQw(3) = 25/96;
        GQw(4) = 25/96;
        
        S=zeros(3,4);
        dS=zeros(2,3,4);
        
        for k =1:4
           sl = GQp(k,1);
           tl = GQp(k,2);
           
           S(1,k) = 1 - sl - tl;
           S(2,k) = sl;
           S(3,k) = tl;
           %ds/dS
           dS(1,1,k) = -1;
           dS(1,2,k) = 1;
           dS(1,3,k) = 0;
           %dt/dS
           dS(2,1,k) = -1;
           dS(2,2,k) = 0;
           dS(2,3,k) = 1;
        end
    elseif nen==4        %QUAD element
       %s-Value                 t-Value
        GQp(1,1) = -sqrt(1/3);  GQp(1,2) = -sqrt(1/3);
        GQp(2,1) = sqrt(1/3);   GQp(2,2) = -sqrt(1/3);
        GQp(3,1) = -sqrt(1/3);  GQp(3,2) = sqrt(1/3);
        GQp(4,1) = sqrt(1/3);   GQp(4,2) = sqrt(1/3);
        
        GQw(1) = 1;
        GQw(2) = 1;
        GQw(3) = 1;
        GQw(4) = 1;
        
        S=zeros(4,4);
        dS=zeros(2,4,4);
        
        for k =1:4
           sl = GQp(k,1);
           tl = GQp(k,2);
           
           S(1,k) = (1/4)*(1-sl)*(1-tl);
           S(2,k) = (1/4)*(1+sl)*(1-tl);
           S(3,k) = (1/4)*(1+sl)*(1+tl);
           S(4,k) = (1/4)*(1-sl)*(1+tl);
           %ds/dS
           dS(1,1,k) = -(1/4)*(1-tl);
           dS(1,2,k) = (1/4)*(1-tl);
           dS(1,3,k) = (1/4)*(1+tl);
           dS(1,4,k) = -(1/4)*(1+tl);
           %dt/dS
           dS(2,1,k) = -(1/4)*(1-sl);
           dS(2,2,k) = -(1/4)*(1+sl);
           dS(2,3,k) = (1/4)*(1+sl);
           dS(2,4,k) = (1/4)*(1-sl);
        end
    end
end


function [K,F] = calculate(nn,ne,enp,nxy,aK,f)
    K = sparse(nn,nn);
    F = sparse(nn,1);
    
    for e=1:ne
        if enp(e,4) == 0%--------------------------------------TRI ELEMENT
            Ke = zeros(3,3);
            Fe = zeros(3,1);
            exy = zeros(3,2);
            
            for i=1:3
                exy(i,1) = nxy(enp(e,i),1);
                exy(i,2) = nxy(enp(e,i),2);
            end
            %GQp loop
            [GQw,S,dS] = defineGQS(3);
            for k=1:4
                Jac(:,:) = dS(:,:,k)*exy(:,:);
                gDS(:,:,k) = inv(Jac)*dS(:,:,k);
                
                for i=1:3
                   for j=1:3
                      Ke(i,j) = Ke(i,j) + (...
                                aK*(gDS(1,i,k)*gDS(1,j,k) +...
                                    gDS(2,i,k)*gDS(2,j,k))...
                                )*det(Jac)*GQw(k);
                   end
                   Fe(i) = Fe(i) + f*S(i,k)*det(Jac)*GQw(k);
                end
                
                
            end
            ix = enp(e,1:3);
            K(ix, ix) = K(ix, ix) + Ke(:,:);
            F(ix) = F(ix) + Fe(:);
        else%-------------------------------------------------QUAD ELEMENT
            Ke = zeros(4,4);
            Fe = zeros(4,1);
            exy = zeros(4,2);
            
            for i=1:4
                index=enp(e,i);
                exy(i,:) = nxy(index,:);
            end
            %GQp loop
            [GQw,S,dS] = defineGQS(4);
            for k=1:4
                Jac(:,:) = dS(:,:,k)*exy(:,:);
                gDSS(:,:,k) = inv(Jac)*dS(:,:,k);
                
                for i=1:4
                   for j=1:4
                      Ke(i,j) = Ke(i,j) + (...
                                aK*(gDSS(1,i,k)*gDSS(1,j,k) +...
                                    gDSS(2,i,k)*gDSS(2,j,k))...
                                )*det(Jac)*GQw(k);
                   end
                   Fe(i) = Fe(i) + f*S(i,k)*det(Jac)*GQw(k);
                end
                
            end
            K(enp(e,:), enp(e,:)) = K(enp(e,:), enp(e,:)) + Ke(:,:);
            F(enp(e,:)) = F(enp(e,:)) + Fe(:);
        end
        
    end
end

function [K,F] = boundCond(bcn,bce,Ki,Fi,enp,nxy)
    nbcn = length(bcn(1,:));
    nbce = length(bce(1,:));
    K = Ki;
    F = Fi;
    %Type 2 BCs
    for i=1:nbce
        e = bce(1,i);
        v = bce(2,i);
        f = bce(3,i);
        
        %Faces
        if enp(e,4) == 0    %TRI ELEMENT
            if f == 3
                n1 = enp(e,f);
                n2 = enp(e,1);
            else 
                n1 = enp(e,f);
                n2 = enp(e,f+1);
            end
        else                %QUAD ELEMENT
            if f == 4
                n1 = enp(e,f);
                n2 = enp(e,1);
            else 
                n1 = enp(e,f);
                n2 = enp(e,f+1);
            end
        end
        dx = nxy(n1,1)-nxy(n2,1);
        dy = nxy(n1,2)-nxy(n2,2);
        d = sqrt(dx^2 + dy^2);
        
        F(n1) = F(n1) + 0.5*v*d;
        F(n2) = F(n2) + 0.5*v*d;
    end
    %Type 1 BCs
    for i=1:nbcn
        ne = bcn(1,i);
        nv = bcn(2,i);
        
        F(ne) = nv;
        K(ne,:) = 0;
        K(ne,ne) = 1;
    end
    
    
end

%POST PROCESSING
function [] = postProcessing(ne,nxy,enp,U)
    
    X = zeros(4,ne);
    Y = zeros(4,ne);
    Z = zeros(4,ne);
    
    for e=1:ne
       if enp(e,4) == 0
           for i=1:3
               X(i,e) = nxy(enp(e,i),1);
               Y(i,e) = nxy(enp(e,i),2);
               Z(i,e) = U(enp(e,i));
           end
           X(4,e) = nxy(enp(e,1),1);
           Y(4,e) = nxy(enp(e,1),2);
           Z(4,e) = U(enp(e,1));
       else
           for i=1:4
               X(i,e) = nxy(enp(e,i),1);
               Y(i,e) = nxy(enp(e,i),2);
               Z(i,e) = U(enp(e,i));
           end
       end
    end
    
    patch(X,Y,Z);
    axis equal;
    colorbar;
    xlabel('x');
    ylabel('y');
    zlabel('u');
end



