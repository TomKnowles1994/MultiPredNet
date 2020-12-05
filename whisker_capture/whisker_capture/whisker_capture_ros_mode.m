
function data = whisker_capture_ros_mode(exec, sim)

% suitable offset to put time stamps in single range
t_offset = 775043392;

% pars
ready = false;

% storage for camera images
eye_camera_images = {{}, {}};
eye_camera_images_n = {[], []};
eye_camera_images_t = {[], []};

% storage for time domain signals
theta_cmd = zeros(24, exec.n_recv, 'single');
theta_meas = zeros(240, exec.n_recv, 'single');
xy_meas = zeros(480, exec.n_recv, 'single');
xy_meas_clean = zeros(480, exec.n_recv, 'single');
protracting = zeros(1, exec.n_recv, 'int8');
neck_meas = zeros(30, exec.n_recv, 'single');
body_pose = zeros(4, 0, 'single');
body_odom = zeros(4, 0, 'single');

% how many received on each channel so far?
n_recv = [0 0 0 0 0 0];

% subscribe
sub = [];
use_bridge_u = 1;
try

	disp('trying to use bridge_u...')
	msginfo = rosmsg('show', 'whiskeye_msgs/bridge_u');
	disp('...OK')
	% if that worked, we can receive bridge_u directly...

catch

	disp('...FAIL')
	use_bridge_u = 0;

end

if use_bridge_u

	% receive bridge_u
	sub.bridge_u = rossubscriber('/whiskeye/head/bridge_u', @callback_bridge_u, 'BufferSize', 5);

else

	% NB: this won't work unless breakout_for_matlab_clients is running
	% already!
	sub.theta = rossubscriber('/whiskeye/head/theta', @callback_theta, 'BufferSize', 5);
	sub.xy = rossubscriber('/whiskeye/head/xy', @callback_xy, 'BufferSize', 5);

end

% these are received in both modes
sub.theta_cmd = rossubscriber('/whiskeye/head/theta_cmd', @callback_theta_cmd, 'BufferSize', 5);
sub.xy_q = rossubscriber('/model/xy_q', @callback_xy_q, 'BufferSize', 5);
sub.prot = rossubscriber('/model/protracting', @callback_prot, 'BufferSize', 5);
if ~sim
	sub.odom = rossubscriber('/whiskeye/body/odom', @callback_odom, 'BufferSize', 5);
    sub.pose = rossubscriber('/body/pose', @callback_pose, 'BufferSize', 5);
else
    sub.pose = rossubscriber('whiskeye/body/pose', @callback_pose, 'BufferSize', 5);
end

% camera signals
get_camera_signals = true;
if get_camera_signals
	sub.cam0 = rossubscriber('/model/cam0/compressed', @callback_cam0, 'BufferSize', 5);
	sub.cam1 = rossubscriber('/model/cam1/compressed', @callback_cam1, 'BufferSize', 5);
end

% publish
if ~isempty(exec.theta_cmd)
	pub_theta = rospublisher('/whiskeye/head/theta_cmd', 'std_msgs/Float32MultiArray');
	msg_theta = rosmessage(pub_theta);
end

% wait
pause(1)

% wait for capture to complete
%utils line
disp('capturing...');
tic
ready = true;
n_sent = 0;
t_timeout = 0;
while any(n_recv < exec.n_recv)
	if n_recv(1) > n_sent
		d = n_recv(1) - n_sent;
		if d > 1
			disp(d)
		end
		n_sent = n_recv(1);

		if ~isempty(exec.theta_cmd)
			msg_theta.Data = single(exec.theta_cmd(:, n_sent));
			send(pub_theta, msg_theta);
		end

		disp([n_recv exec.n_recv])
		t_timeout = toc;
	else
		pause(0.001)
		if t_timeout && exec.t_timeout
			dt = toc - t_timeout;
			if dt > exec.t_timeout
				N = min(n_recv);
				size(theta_cmd)
				theta_cmd = theta_cmd(:, 1:N);
				theta_meas = theta_meas(:, 1:N);
				xy_meas = xy_meas(:, 1:N);
				xy_meas_clean = xy_meas_clean(:, 1:N);
				protracting = protracting(1:N);
				neck_meas = neck_meas(:, 1:N);
				break
			end
		end
	end
%     pause(0.1);
end
ready = false;
toc
%utils line

% clear subscriptions
clear sub

% reshape
theta_cmd = reshape(theta_cmd, 4, 6, []);
theta_meas = reshape(theta_meas, 4, 6, []);
xy_meas = reshape(xy_meas, 2, 4, 6, []);
xy_meas_clean = reshape(xy_meas_clean, 2, 4, 6, []);
neck_meas = reshape(neck_meas, 3, []);

% return results
data = [];
if get_camera_signals
	cam = [];
	cam.n = eye_camera_images_n{1};
	cam.t = eye_camera_images_t{1};
	cam.images = eye_camera_images{1};
	cam(2).n = eye_camera_images_n{2};
	cam(2).t = eye_camera_images_t{2};
	cam(2).images = eye_camera_images{2};
	data.eye_camera_images = cam;
end
data.theta_cmd = theta_cmd;
data.theta_meas = theta_meas;
data.xy_meas = xy_meas;
data.xy_meas_clean = xy_meas_clean;
data.protracting = protracting;
data.neck_meas = neck_meas;
data.body_pose = body_pose;
if ~sim
    data.body_odom = body_odom;
end



    function callback_theta_cmd(~, theta)
        if ready && n_recv(1) < exec.n_recv
            n_recv(1) = n_recv(1) + 1;
            theta_cmd(:, n_recv(1)) = theta.Data;
        end
    end

    function callback_theta_sub(theta)
        if ready && n_recv(2) < exec.n_recv
            n_recv(2) = n_recv(2) + 1;
            theta_meas(:, n_recv(2)) = theta;
        end
    end

    function callback_theta(~, msg)
		callback_theta_sub(msg.Data)
    end

    function callback_xy_sub(xy)
        if ready && n_recv(3) < exec.n_recv
            n_recv(3) = n_recv(3) + 1;
            xy_meas(:, n_recv(3)) = xy;
        end
    end

    function callback_xy(~, msg)
		callback_xy_sub(msg.Data);
    end

    function callback_neck_sub(neck)
        if ready && n_recv(6) < exec.n_recv
            n_recv(6) = n_recv(6) + 1;
            neck_meas(:, n_recv(6)) = neck;
        end
    end

    function callback_bridge_u(~, msg)
		callback_theta_sub(msg.Theta.Data);
		callback_xy_sub(msg.Xy.Data);
		callback_neck_sub(msg.Neck.Data);
    end

    function callback_xy_q(~, msg)
        if ready && n_recv(4) < exec.n_recv
            n_recv(4) = n_recv(4) + 1;
            xy_meas_clean(:, n_recv(4)) = msg.Data;
        end
    end

    function callback_prot(~, msg)
        if ready && n_recv(5) < exec.n_recv
            n_recv(5) = n_recv(5) + 1;
            protracting(n_recv(5)) = msg.Data;
        end
    end

    function callback_pose(~, msg)
		% [x, y, theta, t_capture]
		% NB: t_capture may not be the same as t_create, and may have
		% plateaus in it due to the capturing process
		if ready
			xytt = [msg.X msg.Y msg.Theta toc];
			body_pose(:, end+1) = xytt';
		end
    end

    function callback_odom(~, msg)
		% [dx, dy, dtheta, t_capture, t_create]
		if ready
			t = msg.Header.Stamp;
			t_create = double(t.Sec - t_offset) + double(t.Nsec) * 1e-9;
			% t_create seems not to work on robotino, so I'm not storing
			% it...
			x = msg.Twist.Twist;
			xytt = [x.Linear.X x.Linear.Y x.Angular.Z toc];
			body_odom(:, end+1) = xytt';
		end
    end

    function callback_cam0(~, msg)
		if ready
			try
				t = toc;
				n = n_recv(1);
				im = msg.readImage();
				eye_camera_images{1}{end+1} = im;
				eye_camera_images_n{1}(end+1) = n;
				eye_camera_images_t{1}(end+1) = t;
			end
		end
    end

    function callback_cam1(~, msg)
		if ready
			try
				t = toc;
				n = n_recv(1);
				im = msg.readImage();
				eye_camera_images{2}{end+1} = im;
				eye_camera_images_n{2}(end+1) = n;
				eye_camera_images_t{2}(end+1) = t;
			end
		end
    end

end
