%%%%%%%% Original Transfer Function Realization %%%%%%%%

num = {6.3701, 0; 0, -9.6547};
den = {[12.9884 1], [1 1]; [1 1], [11.1996 1]};

sys1 = tf(num, den, 'IODelay', [0.7778 0; 0 2.3333]);

%%%%%%%% Converting Transfer Functions to State Space %%%%%%%%

G11 = tf([0 6.3701], [12.9884 1]);
G12 = tf([0 0], [1 1]);
G21 = tf([0 0], [1 1]);
G22 = tf([0 -9.6547], [11.1996 1]);

G = [G11, G12; G21, G22];
[A, B, C, D] = ssdata(ss(G, 'min'));

sys2 = ss(A, B, C(1, :), D(1, :), 'InputDelay', [0.7778, 0]);
sys3 = ss(A, B, C(2, :), D(2, :), 'InputDelay', [0, 2.3333]);

%%%%%%%% Simulation Parameters %%%%%%%%
t = 0:1:150;
% u = ones(1, numel(t));
u = zeros(1, numel(t));
u(1, :) = 15.7;
u(2, :) = 0;
x0 = [51, -58];

%%%%%%%% Simulation of y1 %%%%%%%%
[y1, x1] = lsim(sys2, u, t, x0);

%%%%%%%% Simulation of y2 %%%%%%%%
[y2, x2] = lsim(sys3, u, t, x0);

%%%%%%%% Simulation of Transfer Function %%%%%%%%
[y, x] = lsim(sys1, u, t, x0);

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

