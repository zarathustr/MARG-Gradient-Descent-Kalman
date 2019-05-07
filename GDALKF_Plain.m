% Gradient-Descent-Algorithm Linear Kalman Filter (GDALKF) by Jin Wu
%
% This is the Plain version of GDALKF
%
% Author: Jin Wu
% e-mail: jin_wu_uestc@hotmail.com
% Copyright 2017



function [xk, Pk] = GDALKF_Plain(...
                        Gyroscope, Accelerometer, Magnetometer, ...
                        Sigma_g, Sigma_a, Sigma_m,...
                        xk_1, Pk_1, ...
                        mu, Mr, dt)
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
    
    mN = Mr(1);               mD = Mr(3);
    

    yk = [
        q0 - mu * (4 * q0 - 2 * mz * mD * q0 - 2 * my * mD * q1 + 2 * mx * mD * q2 - 2 * mx * mN * q0 - 2 * mz * mN * q2 + 2 * my * mN * q3 - 2 * az * q0 - 2 * ay * q1 + 2 * ax * q2);
        q1 - mu * (4 * q1 - 2 * my * mD * q0 + 2 * mz * mD * q1 - 2 * mx * mD * q3 - 2 * mx * mN * q1 - 2 * my * mN * q2 - 2 * mz * mN * q3 - 2 * ay * q0 + 2 * az * q1 - 2 * ax * q3);
        q2 - mu * (4 * q2 + 2 * mx * mD * q0 + 2 * mz * mD * q2 - 2 * my * mD * q3 - 2 * mz * mN * q0 - 2 * my * mN * q1 + 2 * mx * mN * q2 + 2 * ax * q0 + 2 * az * q2 - 2 * ay * q3);
        q3 - mu * (4 * q3 - 2 * mx * mD * q1 - 2 * my * mD * q2 - 2 * mz * mD * q3 + 2 * my * mN * q0 - 2 * mz * mN * q1 + 2 * mx * mN * q3 - 2 * ax * q1 - 2 * ay * q2 - 2 * az * q3)];


    Ham = [
        - 2 * mu * q2, 2 * mu * q1,   2 * mu * q0, - mu * (2 * mD * q2 - 2 * mN * q0), mu * (2 * mD * q1 - 2 * mN * q3),   mu * (2 * mD * q0 + 2 * mN * q2),
          2 * mu * q3, 2 * mu * q0, - 2 * mu * q1,   mu * (2 * mD * q3 + 2 * mN * q1), mu * (2 * mD * q0 + 2 * mN * q2), - mu * (2 * mD * q1 - 2 * mN * q3),
        - 2 * mu * q0, 2 * mu * q3, - 2 * mu * q2, - mu * (2 * mD * q0 + 2 * mN * q2), mu * (2 * mD * q3 + 2 * mN * q1), - mu * (2 * mD * q2 - 2 * mN * q0),
          2 * mu * q1, 2 * mu * q2,   2 * mu * q3,   mu * (2 * mD * q1 - 2 * mN * q3), mu * (2 * mD * q2 - 2 * mN * q0),   mu * (2 * mD * q3 + 2 * mN * q1)
    ];

    Eps = Ham * [Sigma_a,    zeros(3, 3);
                 zeros(3, 3),    Sigma_m] * Ham';

    x_ = Phi * xk_1;
    Pk_ = Phi * Pk_1 * Phi' + Xi;
    Gk = Pk_ * (inv(Pk_ + Eps));
    Pk = (eye(4) - Gk) * Pk_;
    xk = x_ + Gk * (yk - x_);
    
    xk = xk ./ norm(xk);

end
