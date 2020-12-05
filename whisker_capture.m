
function data = whisker_capture(theta_cmd, use_ros_mode, sim)

% argument missing
if ~exist('theta_cmd', 'var')
	theta_cmd = zeros(24, 50);
end

% argument missing
if ~exist('use_ros_mode', 'var')
	use_ros_mode = false;
end

% argument missing
if ~exist('sim', 'var')
	sim = false;
end

% create default execution
exec = [];
exec.fS = 50;
exec.theta_cmd = []; % if non-empty, receive whilst sending this command
exec.t_timeout = 0; % if non-zero, receive until timeout
exec.n_recv = 0; % if non-zero, receive fixed number of samples

% handle argument is just a time to run for
if isscalar(theta_cmd)
	if theta_cmd > 0
		tf = theta_cmd;
		exec.n_recv = tf * exec.fS;
	else
		% t = 0 means "run until data runs out" which is useful for reading
		% back results from rosbags
 		exec.n_recv = 60 * 25 * exec.fS; % space for so many minutes
		exec.t_timeout = 5; % seconds to timeout
	end
else
	exec.theta_cmd = theta_cmd;
	exec.n_recv = size(exec.theta_cmd, 2);
end


% delegate
if use_ros_mode
	data = whisker_capture_ros_mode(exec, sim);
else
	data = whisker_capture_daemon_mode(exec);
end

% augment results
if ~isfield(data, 'theta_cmd')
	data.theta_cmd = reshape(exec.theta_cmd, 4, 6, []);
end
data.t_now = now;
data.timestamp = datestr(data.t_now, 'yymmdd-HHMMSS');

% quick display
if nargout == 0

	% display
	tt = (1:(exec.n_recv*10))/500;
	t_offset_cmd_meas = 0 / 50;

	% display
	selectfigure whisker_capture_theta
	clf
	p = panel();
	p.pack(4, 6);
	p.margin = [8 8 5 5];
	ax = [0 max(tt) -75 75];
	for r = 1:6
		for c = 1:4
			p(c, r).select();
			tt_ = tt(10:10:end);
			if isempty(exec.theta_cmd)
				th = NaN(size(tt_));
			else
				th = squeeze(exec.theta_cmd(c, r, :));
			end
			plot(tt_, th / pi * 180)
			hold on
			th = squeeze(data.theta_meas(c, r, :));
			plot(tt - t_offset_cmd_meas, th / pi * 180);
			axis(ax)
		end
	end

	% display
	selectfigure whisker_capture_xy
	clf
	p = panel();
	p.pack(4, 6);
	p.margin = [8 8 5 5];
	ax = [0 max(tt) -1 1];
	for r = 1:6
		for c = 1:4
			p(c, r).select();
			xy = squeeze(data.xy_meas(:, c, r, :))';
			plot(tt, xy);
			axis(ax)
		end
	end

end
