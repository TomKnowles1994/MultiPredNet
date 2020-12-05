
% load the data file first, if available, or use the code below to capture
% the data from the rosbag
% if ~exist('data', 'var')
% 	if exist('/tmp/data', 'file')
% 		load /tmp/data
% 	end
% end

%% import from rosbag
if ~exist('data', 'var')
	addpath ../whisker_capture
	% whisker_capture(theta_cmd, use_ros_mode, sim)
	% If data capture from simulator (sim = true) there will be no body odometry
	data = whisker_capture(0, true, true);
end
