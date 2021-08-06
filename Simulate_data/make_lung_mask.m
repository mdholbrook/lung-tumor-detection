function [X_crop, lung_mask, lung_mask_large] = make_lung_mask(X, crop_buffer)

fprintf('Generating lung mask\n');

% Crop the image to remove air before and after the animal in all dims
fprintf('\tCropping image to remove empty air\n');

tic;
sz = size(X);
dinds = 1:length(sz);
threshold = [0.1, 0.1, 0.5];
top_offset = 10;
X_crop = X(:,:,top_offset:end);
crop_list = zeros(2,3);

clearvars X;

% Initial cropping to reduce computation time
for z = 1:length(sz)
    
    % Permute axis
    dinds1 = circshift(dinds, z);
    X_crop = permute(X_crop, dinds1);
    
    % Calculate start and stop indices
    zsum = squeeze(sum(sum(X_crop + 1000, 1), 2));

    start_ind = find(zsum > threshold(z)*max(zsum), 1, 'first');
    end_ind = find(zsum > threshold(z)*max(zsum), 1, 'last');
    
    % Save crop indices
    crop_list(:,dinds1(end)) = [start_ind; end_ind];
    
    % Crop image
    X_crop = X_crop(:, :, start_ind:end_ind);
    
    % Invert purmute
    X_crop = ipermute(X_crop, dinds1);   
    
end

crop_list(:,end) = crop_list(:,end) + top_offset - 1;


%% Generate lung mask

% Threshold image
thresh = -100;
lung_mask = X_crop < thresh;

% Compute continous labels
fprintf('\tFinding air regions\n')
labels = bwlabeln(lung_mask);

% Remove background
bkgrnd_label = labels(1, 1, 1);
lung_mask(bkgrnd_label == labels) = 0;

% Remove other air pockets
fprintf('\tRemoving air pockets and sorting labels\n')
[counts, edges] = histcounts(labels, unique(labels(:)) - 0.5);

[vals, ind] = sort(counts, 'descend');

fprintf('\tFiltering label results\n')
flag = true;
largest_segs = 4;
segs = zeros(largest_segs, 1); % segmentation value
com = zeros(largest_segs, 1);  % center of mass
seg_count = zeros(largest_segs, 1);  % counts for each label
G = zeros(largest_segs, 1);  % sum of gradients
com_weight = 1;
z = 1;
n = 1;
while flag
    
    seg_val = round(edges(ind(z)) + 0.5);
    
    if seg_val ~= 0 && seg_val ~= bkgrnd_label
        
        
        % Get label mask
        tmp_seg = labels == seg_val;
        
        % Get segmentation value and count
        segs(n) = seg_val;
        seg_count(n) = vals(z);
        
        % Get the number of voxels in each label per slice
        zsums = squeeze(sum(sum(tmp_seg, 1), 2));
        
        % Find center of mass in z-direction
        com(n) = sum(zsums .* [1:length(zsums)]') ./ sum(zsums);
        
        % Find the surface area of each mask
        [Gx, Gy, Gz] = imgradientxyz(tmp_seg);
        
        Gim = sqrt(Gx.^2 + Gy.^2 + Gz.^2);
        G(n) = sum(Gim(:) > 0.5 );
        
        n = n + 1;
        
        if n > largest_segs
            
            flag = false;
            
        end
        
    end
    
    z = z + 1;
    
end

% Sanity check for size
ref_vol = 2375000;
inds = (0.2 * ref_vol) < seg_count & seg_count < (2 * ref_vol);
seg_count(~inds) = 0;

% Cluster the found points
prob_voxs = zeros(1,largest_segs);
prob_C = zeros(3,largest_segs);
for z = 1:largest_segs
    % Compute COM for 2 clusters
    com_rand = [round(size(labels,1)/2) + 30, round(size(labels,2)/2), com(z);
                round(size(labels,1)/2) - 30, round(size(labels,2)/2), com(z)];
    [c, voxs] = compute_clusters(labels == segs(z), com_rand);
    
    % Get voxel probability
    norm_voxs = voxs / sum(voxs);
    prob_voxs(z) = min(norm_voxs);
    
    % Get COM probabilities - maximize x,y differences, minimize z diffs
    prob_C(1, z) = abs(c(1,1) - c(2,1)) / size(labels, 1);
    prob_C(2, z) = abs(c(1,2) - c(2,2)) / size(labels, 2);
    prob_C(3, z) = 1 - abs(c(1,3) - c(2,3)) / size(labels, 3);
    
end

% Convert to a single probability by multiplying probs
prob_C = prod(prob_C);
prob_C = prob_C(:) / max(prob_C);

% Condition for mean lung value
for z = 1:largest_segs
    mval(z) = mean(X_crop(labels == segs(z)));
    inds(z) = -600 < mval(z);
end
mvals = inds;

% Normalize the counts
seg_count = seg_count / sum(seg_count);
seg_thresh = seg_count > 0.25;
com = 1 - com / size(X_crop, 3);
G = G / sum(G);

% Compute probability
% Voxel count, center of mass axial location, surface area, grey values,
% clustering COM
lung_prob = seg_count .* (com_weight * com) .* G .* mvals .* prob_C;

% Get argmax
[~, maxind] = max(lung_prob);

% Get lung segmenation value
seg_val = segs(maxind);
  
% Update lung mask
lung_mask(labels ~= seg_val) = 0;

% Dilate mask
fprintf('\tClosing the mask\n');
SE = strel('sphere', 5);
lung_mask = imclose( lung_mask, SE);

% Create full-sized mask
lung_mask_large = zeros(sz);
lung_mask_large(crop_list(1,1):crop_list(2,1),...
                crop_list(1,2):crop_list(2,2),...
                crop_list(1,3):crop_list(2,3)) = lung_mask;

% Cropping around mask
fprintf('\tCropping around the lung mask\n');
[X_crop, lung_mask] = crop_image_to_mask(X_crop, lung_mask, crop_buffer);

% Convert all outputs to single
lung_mask = single(lung_mask);
lung_mask_large = single(lung_mask_large);

fprintf('\tDone making mask!\n\n');
fprintf('\t\tTook %0.2f seconds\n', toc);

end



function [X_out, mask_out] = crop_image_to_mask(X, mask, buffer)
% Crops an input image and mask to the size of the mask + buffer
% Args:
%   X: input image
%   mask: input mask
%   buffer: required distance between mask edge and image edge
% Returns:
%   X_out: cropped image
%   mask_out: cropped mask

% Account for trachea
zsums = squeeze(sum(sum(mask,1),2));
ind = find(zsums > 0.04*max(zsums), 1, 'first');

if ind > buffer
     ind = ind - buffer;
end

% Remove trachea
X = X(:,:,ind:end);
mask = mask(:,:,ind:end);

% Set up variables
X_out = X;
mask_out = mask;
axes = 1:3;
crop_list = zeros(2,3);

for z = 1:3
    
    % Permute mask
    axes = circshift(axes, 1);
    mask_r = permute(mask, axes);
    
    % Get total mask counts along 3rd axis
    sums = squeeze(sum(sum(mask_r, 1), 2));
    
    % Find locations where there are positive values
    sums_locs = find(sums > 0.9);
    
    % Assign start and stop values
    start = sums_locs(1) - buffer;
    stop = sums_locs(end) + buffer;
    
    % Account for image boundaries
    if start < 1
        start = 1;
    end
    
    if stop > length(sums)
        stop = length(sums);
    end
    
    % Save start and stop indices
    crop_list(:, z) = [start; stop];
    
    % Update output images
    mask_out = permute(mask_out, axes);
    X_out = permute(X_out, axes);
    
    mask_out = mask_out(:, :, start:stop);
    X_out = X_out(:, :, start:stop);
    
    mask_out = ipermute(mask_out, axes);
    X_out = ipermute(X_out, axes);
    
end


end


function [C, voxs] = compute_clusters(X, com_rand)

% Get indices of positive values
[a, b, c] = ind2sub(size(X), find(X==1));
inds = [a, b, c]; % N by 3 array 
 
% Set up indices as a GPU array
% inds_ = gpuArray(inds);
% com_rand_ = gpuArray(com_rand);

% Cluster points
k = 2;
[idx, C] = kmeans(inds, k, 'Start', com_rand, 'Options', statset('UseParallel',1));

% Get center-of-cluster locations
% C = gather(C_);

% Get number of voxels in each cluster
% voxs_ = gpuArray(zeros(1,k));
voxs = zeros(1,k);
for z = 1:k
    voxs(z) = sum(idx == z);
end

end


function [C, voxs] = compute_clusters_gpu(X, com_rand)

% Get indices of positive values
[a, b, c] = ind2sub(size(X), find(X==1));
inds = [a, b, c]; % N by 3 array 
 
% Set up indices as a GPU array
inds_ = gpuArray(inds);
com_rand_ = gpuArray(com_rand);

% Cluster points
k = 2;
[idx_, C_] = kmeans(inds_, k, 'Start', com_rand_);

% Get center-of-cluster locations
C = gather(C_);

% Get number of voxels in each cluster
voxs_ = gpuArray(zeros(1,k));
for z = 1:k
    voxs_(z) = sum(idx_ == z);
end

voxs = gather(voxs_);

end
