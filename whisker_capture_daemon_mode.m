
function data = whisker_capture_daemon_mode(exec)

% storage
theta_meas = zeros(240, exec.n_recv, 'single');
xy_meas = zeros(480, exec.n_recv, 'single');


%%%% DAEMON

% interf
file_in = '~/device/whisker_exec.in';
file_out = '~/device/whisker_exec.out';

% clean
if exist(file_out, 'file')
	delete(file_out);
end

% encode and send
disp('send input...')
json = jsonencode(exec.theta_cmd);
fid = fopen(file_in, 'w');
fwrite(fid, json);
fclose(fid);
disp('...OK')

% wait for response
disp('wait for output...')
while 1
	if exist(file_out, 'file')
		break
	end
	pause(0.1)
end
disp('...OK')

% load it and delete it
lines = textfile(file_out, 'a');
delete(file_out);

% decode it
data = [];

% theta_cmd
data.theta_cmd = reshape(exec.theta_cmd, 4, 6, []);

% theta_meas
x = jsondecode(lines{1});
x = reshape(x, 10, 6, 4, []);
x = permute(x, [3 2 1 4]);
x = reshape(x, 4, 6, []);
data.theta_meas = x;

% xy_meas
x = jsondecode(lines{2});
x = reshape(x, 10, 6, 4, 2, []);
x = permute(x, [4 3 2 1 5]);
x = reshape(x, 2, 4, 6, []);
data.xy_meas = x;

% xy_meas_clean
x = jsondecode(lines{3});
x = reshape(x, 10, 6, 4, 2, []);
x = permute(x, [4 3 2 1 5]);
x = reshape(x, 2, 4, 6, []);
data.xy_meas_clean = x;
