function [ pose_dist_norm, reps_dist_norm, rand_reps_dist_norm, r, r_rand ] = Calc_RDM(pose, reps, rot_scale)
%%%
% Returns: 
% Dis-similarity matrix of pose from dataset
% Dis-similarity matrix of learned representations from dataset
% Spearman's rank coefficients between pose and representation similarities
% Spearmans rank coeffeicients for random representations for a reference
%
% Aurguments:
% poses and reps of the same length
% a scaling factor for contribution of orientation change to pose distance
%%%

if length(reps(:,1))~=length(pose(:,1)) 
    fprintf('poses and reps different sizes')
    return
end
N=length(reps(:,1));        % number of data points

% prep containers
pose_dist           = zeros(N,N);
pose_dist_norm      = zeros(N,N);
reps_dist           = zeros(N,N);
reps_dist_norm      = zeros(N,N);
rand_reps           = rand(N,length(reps(2,:))); % normalised random representations
rand_reps_dist_norm = zeros(N,N);


r = zeros(N,1);
r_h = zeros(N,1);
t = zeros(N,1);
t_h = zeros(N,1);
p = zeros(N,1);

p_h = zeros(N,1);
r_rand = zeros(N,1);
t_rand = zeros(N,1);
p_rand = zeros(N,1);

pos_scale = 1; % scaling for position distance

for i=1:N       % cycle through all data points
     %% build dissimilarity matrices between each data point to all others
     for j=1:N
        pose_dist(i,j) = rot_scale*(norm(wrapToPi(pose(i,3)-pose(j,3)))) + pos_scale*(norm(pose(i,1:2)-pose(j,1:2)));
        reps_dist(i,j) = 1 - corr(reps(i,:)' , reps(j,:)' , 'type', 'Pearson');         % 1- correlation representation space
        rand_reps_dist_norm(i,j) = 1 - corr(rand_reps(i,:)' , rand_reps(j,:)' , 'type', 'Pearson'); % 1- correlation random baseline
     end 
end
% normalise
pose_dist_norm = pose_dist / max(max(pose_dist));                           
reps_dist_norm = reps_dist ; %/ max(max(reps_dist)); 
rand_reps_dist_norm = rand_reps_dist_norm ; %/ max(max(rand_reps_dist_norm)); 

for i=1:N    
    %spearman's rank correlation
    [r(i) t(i) p(i)] = spear(pose_dist_norm(i,:)', reps_dist_norm(i,:)');
    [r_rand(i) t_rand(i) p_rand(i)] = spear(pose_dist_norm(i,:)', rand_reps_dist_norm(i,:)');
end

