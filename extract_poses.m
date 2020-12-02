function [pose_return odom_return] = extract_poses(data, sim) 


% argument missing
if ~exist('sim', 'var')
	sim = false;
end

% %% recover body pose from body odometry and head offset from neck odometry
 recover_head_offset;
 if ~sim
    recover_body_pose;
 end

%% resize body_pose and head_offset and add offset to pose
for i=1:3
    head(i,:)=decimate(cast(data.head_offset(i,:),'double'),10);
    pose(i,:)=interp1((1:length(data.body_pose(1,:))), data.body_pose(i,:), linspace(1,length(data.body_pose(1,:)),length(head(i,:))));
    pose_head(i,:)=pose(i,:)+head(i,:);
    if ~sim
        odom(i,:)=interp1((1:length(data.body_pose_est_from_odom(1,:))), data.body_pose_est_from_odom(i,:), linspace(1,length(data.body_pose_est_from_odom(1,:)),length(head(i,:))));
        odom_head(i,:)=odom(i,:)+head(i,:);
    else
        odom_head(i,:)=pose_head(i,:); % no odometry error from simulation
    end 
end

pose_head(3,:)= wrapToPi(pose_head(3,:));
odom_head(3,:)= wrapToPi(odom_head(3,:));

%% find elements in pose that coincide with camerea frames
f = find(data.protracting(2:end) == 0 & data.protracting(1:end-1) == 1);
for i=1:length(f)
    pose_return(i,:)=pose_head(:,f(i));
    odom_return(i,:)=odom_head(:,f(i));
end