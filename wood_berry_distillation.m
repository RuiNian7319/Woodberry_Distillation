%%%%%%%% Original Transfer Function Realization %%%%%%%%

num = {12.8, -18.9; 6.6, -19.4};
den = {[16.7 1], [21, 1]; [10.9, 1], [14.4, 1]};

sys1 = tf(num, den, 'IODelay', [1 3; 7 3]);

%%%%%%%% Converting Transfer Functions to State Space %%%%%%%%

G11 = tf([0 12.8], [16.7 1]);
G12 = tf([0 -18.9], [21 1]);
G21 = tf([0 6.6], [10.9 1]);
G22 = tf([0 -19.4], [14.4 1]);

G = [G11, G12; G21, G22];
[A, B, C, D] = ssdata(ss(G, 'min'));

sys2 = ss(A, B, C(1, :), D(1, :), 'InputDelay', [1, 3]);
sys3 = ss(A, B, C(2, :), D(2, :), 'InputDelay', [7, 3]);

%%%%%%%% Simulation Parameters %%%%%%%%
t = 0:1:150;
% u = ones(2, numel(t));
u = zeros(2, numel(t));
u(1, :) = 0.157;
u(2, :) = 0.05337;
x0 = [0, 0, 0, 0];

%%%%%%%% Simulation of y1 %%%%%%%%
[y1, x1] = lsim(sys2, u, t, x0);
% figure()
% plot(t, y1, t, u1)
% legend('Response', 'Input')

%%%%%%%% Simulation of y2 %%%%%%%%
[y2, x2] = lsim(sys3, u, t, x0);
% figure();
% plot(t, y2, t, u2)
% legend('Response', 'Input')

%%%%%%%% Simulation of Transfer Function %%%%%%%%
[y, x] = lsim(sys1, u, t, x0);
% figure();
% plot(t, y, t, u)
% legend('Response', 'Input')

%%%%%%%%  Figure Generation  %%%%%%%%
figure()
plot(t, y(:, 1), t, y1, '--r')
legend('Transfer Function', 'State Space', 'Interpreter', 'latex');
xlabel('Time, \textit{t} (s)', 'Interpreter', 'latex');
ylabel('\%MeOH in Distillate, \textit{$X_D$} (\%)', 'Interpreter', 'latex');

figure()
plot(t, y(:, 2), t, y2, '--r')
legend('Transfer Function', 'State Space', 'Interpreter', 'latex');
xlabel('Time, \textit{t} (s)', 'Interpreter', 'latex');
ylabel('\%MeOH in Bottoms, \textit{$X_B$} (\%)', 'Interpreter', 'latex');

%%%%%%%% Simulation of the SISO System %%%%%%%%
% num = [0 1];
% den = [10 1];
% sys4 = tf(num, den, 'IODelay', 0.8);

%%%%%%%% Step test of the systems %%%%%%%%
% step(sys1);
% figure()
% step(sys2);
% figure()
% step(sys3);