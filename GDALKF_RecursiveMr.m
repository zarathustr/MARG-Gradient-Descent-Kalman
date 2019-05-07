% Gradient-Descent-Algorithm Linear Kalman Filter (GDALKF) by Jin Wu
%
% This is the Recursive Magnetometer Reference Vector version of GDALKF
%
% Author: Jin Wu
% e-mail: jin_wu_uestc@hotmail.com
% Copyright 2017



function [xk, Pk, mu, yk] = GDALKF_RecursiveMr(...
                        Gyroscope, Accelerometer, Magnetometer, ...
                        Sigma_g, Sigma_a, Sigma_m, ...
                        xk_1, Pk_1, ...
                        mu, dt, adaptive)

    acc = Accelerometer;      acc = acc ./ norm(acc);
    mag = Magnetometer;       mag = mag ./ norm(mag);

    wx = Gyroscope(1);        wy = Gyroscope(2);        wz = Gyroscope(3);
    ax = acc(1);              ay = acc(2);              az = acc(3);
    mx = mag(1);              my = mag(2);              mz = mag(3);
    
    q0 = xk_1(1);             q1 = xk_1(2);             q2 = xk_1(3);         q3 = xk_1(4);
    
    
    omega4 = [  0, -wx, -wy, -wz;
               wx,   0,  wz, -wy;
               wy, -wz,   0,  wx;
               wz,  wy, -wx,   0];
   
    Phi = eye(4) + dt / 2 * omega4;
    
    Dk = [  q1,  q2,  q3;
           -q0, -q3, -q2;
            q2, -q0, -q1;
           -q2,  q1, -q0];
       
    Xi = dt * dt / 4 * Dk * Sigma_g * Dk';
    
    mN = ((mx * (q0^2 + q1^2 - q2^2 - q3^2) - ...
           my * (2 * q0 * q3 - 2 * q1 * q2) + ...
           mz * (2 * q0 * q2 + 2 * q1 * q3))^2 + ...
          (my * (q0^2 - q1^2 + q2^2 - q3^2) + ...
           mx * (2 * q0 * q3 + 2 * q1 * q2) - ...
           mz * (2 * q0 * q1 - 2 * q2 * q3))^2)^0.5;
    mD = mz * (q0^2 - q1^2 - q2^2 + q3^2) - ...
         mx * (2 * q0 * q2 - 2 * q1 * q3) + ...
         my * (2 * q0 * q1 + 2 * q2 * q3);
    
    
    if(adaptive == 1)
        if(mu == 0)
            
            Kam_tilde = [ 
                2 * az + 2 * mD * mz + 2 * mN * mx - 4,                    2 * ay + 2 * mD * my,       2 * mN * mz - 2 * mD * mx - 2 * ax,                           -2 * mN * my;
                                  2 * ay + 2 * mD * my,  2 * mN * mx - 2 * mD * mz - 2 * az - 4,                              2 * mN * my,     2 * ax + 2 * mD * mx + 2 * mN * mz;
                    2 * mN * mz - 2 * mD * mx - 2 * ax,                             2 * mN * my, - 2 * az - 2 * mD * mz - 2 * mN * mx - 4,                   2 * ay + 2 * mD * my;
                                          -2 * mN * my,      2 * ax + 2 * mD * mx + 2 * mN * mz,                     2 * ay + 2 * mD * my, 2 * az + 2 * mD * mz - 2 * mN * mx - 4];
        
            x = Kam_tilde * xk_1;
            y = dt / 2 * omega4 * xk_1;
            
            tau1 = x' * x;
            tau2 = - x' * y - y' * x;

            mu = - tau2 / tau1 / 2;
        else
            alpha = ax * mx + ay * my + az * mz;
            g = 0.8;
            V = mN * sqrt(1 - alpha^2);
            mu = (1 - sqrt((1 - g)^2 + g^2 + 2 * g * (1 - g) * (alpha * mD + mN * sqrt(1 - alpha^2)))) / (4 - 2 * sqrt(2 + 2 * alpha * mD + 2 * V));
        end
    end

    yk = [
        q0 - mu * (4 * q0 - 2 * mz * mD * q0 - 2 * my * mD * q1 + 2 * mx * mD * q2 - 2 * mx * mN * q0 - 2 * mz * mN * q2 + 2 * my * mN * q3 - 2 * az * q0 - 2 * ay * q1 + 2 * ax * q2);
        q1 - mu * (4 * q1 - 2 * my * mD * q0 + 2 * mz * mD * q1 - 2 * mx * mD * q3 - 2 * mx * mN * q1 - 2 * my * mN * q2 - 2 * mz * mN * q3 - 2 * ay * q0 + 2 * az * q1 - 2 * ax * q3);
        q2 - mu * (4 * q2 + 2 * mx * mD * q0 + 2 * mz * mD * q2 - 2 * my * mD * q3 - 2 * mz * mN * q0 - 2 * my * mN * q1 + 2 * mx * mN * q2 + 2 * ax * q0 + 2 * az * q2 - 2 * ay * q3);
        q3 - mu * (4 * q3 - 2 * mx * mD * q1 - 2 * my * mD * q2 - 2 * mz * mD * q3 + 2 * my * mN * q0 - 2 * mz * mN * q1 + 2 * mx * mN * q3 - 2 * ax * q1 - 2 * ay * q2 - 2 * az * q3)];

    FmN = zeros(3, 3);
    FmD = zeros(3, 3);

    for i = 1 : 3
        for j = 1 : 3
            
            C = q2c(xk_1);
            
            UmNj = (C(j, 1) * (C(1, 1) * mx + C(2, 1) * my + C(3, 1) * mz)+ ...
                    C(j, 2) * (C(1, 2) * mx + C(2, 2) * my + C(3, 2) * mz)) / mN;
              
            UmDj = C(j, 3);
            
            if(i == j)
                FmN(i, j) = mN + mag(i) * UmNj;
                FmD(i, j) = mD + mag(i) * UmDj;
            else
                FmN(i, j) = mag(i) * UmNj;
                FmD(i, j) = mag(i) * UmDj;
            end
        end
    end
    
    Lambda1 = [
        - q2,   q1,   q0;
          q3,   q0, - q1;
        - q0,   q3, - q2;
          q1,   q2,   q3];
      
    Lambda2 = [
          q0, - q3,   q2;
          q1,   q2,   q3;
        - q2,   q1,   q0;
        - q3,  -q0,   q1];

    Eps = 4 * mu^2 * (Lambda1 * Sigma_a * Lambda1'+ ...
                      Lambda2 * FmN * Sigma_m * FmN' * Lambda2'+ ...
                      Lambda1 * FmD * Sigma_m * FmD' * Lambda1');

    x_ = Phi * xk_1;
    Pk_ = Phi * Pk_1 * Phi' + Xi;
    Gk = Pk_ * (inv(Pk_ + Eps));
    Pk = (eye(4) - Gk) * Pk_;
    xk = x_ + Gk * (yk - x_);
    
    xk = xk ./ norm(xk);
    yk = yk ./ norm(yk);

end
