function [coord, lung_mask] = loc_in_mask_v2( lung_mask, tumor_mask, overlap )
%loc_in_mask: returns the location for a tumor within a mask. 
% v2: now computes overlap via convolution and only picks valid points
% rather than trial and error.
% Args:
%   mask: 3D mask representing the lungs
%   tumor_mask: mask of the tumor
%   overlap: the percent overlap of the lung and tumor required to keep

% Setup
sz = size(lung_mask);
szt = size(tumor_mask);

%% Create a cropped lung mask to speed up calculations
buffer = 20; % ceil(max(sz)/2);

% Account for trachea
zsums = squeeze(sum(sum(lung_mask,1),2));
ind = find(zsums > 0.04*max(zsums), 1, 'first');

if ind > buffer
     ind = ind - buffer;
end

% Remove trachea
lung_mask_crop = lung_mask(:,:,ind:end);

% Set up variables
lung_crop = lung_mask_crop;
axes = 1:3;

starts = zeros(1, 3);
stops = zeros(1, 3);

for z = 1:3
    
    % Permute mask
    axes = circshift(axes, 1);
    mask_r = permute(lung_mask_crop, axes);
    
    % Get total mask counts along 3rd axis
    sums = squeeze(sum(mask_r, [1, 2]));
    
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
    
    starts(axes(3)) = start;
    stops(axes(3)) = stop;
    
    % Reorder axes
    lung_crop = permute(lung_crop, axes);
    
    % Crop
    lung_crop = lung_crop(:, :, start:stop);
    
    % Reverse permute operation
    lung_crop = ipermute(lung_crop, axes);
    
end

% Correct z index
starts(3) = starts(3) + ind;
stops(3) = stops(3) + ind;

%% Crop the tumor image
tu_starts(1) = find(squeeze(sum(tumor_mask, [2, 3])) > 0, 1, 'first');
tu_stops(1) = find(squeeze(sum(tumor_mask, [2, 3])) > 0, 1, 'last');
tu_starts(2) = find(squeeze(sum(tumor_mask, [1, 3])) > 0, 1, 'first');
tu_stops(2) = find(squeeze(sum(tumor_mask, [1, 3])) > 0, 1, 'last');
tu_starts(3) = find(squeeze(sum(tumor_mask, [1, 2])) > 0, 1, 'first');
tu_stops(3) = find(squeeze(sum(tumor_mask, [1, 2])) > 0, 1, 'last');

tumor_crop = tumor_mask(tu_starts(1):tu_stops(1),...
                             tu_starts(2):tu_stops(2),...
                             tu_starts(3):tu_stops(3));
                         
% Get size of cropped vols
sz_c = size(lung_crop);
szt_c = size(tumor_crop);

%% Compute possible areas for the tumor
% Overlap
overlap_thresh = sum(tumor_mask(:)) * overlap;

% Create tumor overlap map with lungs
overlap_map = gather(convn(gpuArray(single(lung_crop)), gpuArray(single(tumor_crop)), 'same'));

% Create thresholded image from which to pick tumor location
lung_mask_thr = overlap_map > overlap_thresh;

% Get list of points in mask
index = find(lung_mask_thr);

% Pick a new tumor if no locations are found
if isempty(index)
    coord = false;
    return
end

% Make weights to favor lung boundaries
mask_edges = zeros(size(lung_mask_thr));
for z = 1:size(mask_edges, 3)
    mask_edges(:, :, z) = edge(lung_mask_thr(:, :, z));
end
w = bwdist(mask_edges);
w = noncircshift(w, round(szt_c/2));
w = 1/(w+0.5);
w = w(index);

%% Pick tumor location

flag = true;
n = 0;
n2 = 0;
while flag
    
    % Select a random point within the volume
    ind_flag = true;
    while ind_flag
        rind = randsample(length(index),1,true,w(:));
        select = index(rind);
        [x, y, z] = ind2sub(size(lung_crop), select);
        
        % Convert points to be centered on tumor volume
        x = floor(x - szt_c(1)/2) + starts(1);
        y = floor(y - szt_c(2)/2) + starts(2);
        z = floor(z - szt_c(3)/2) + starts(3);

        % Account for out-of-range points
        if ((x + szt_c(1)/2) < sz(1) && (y + szt_c(2)/2) < sz(2) && (z + szt_c(3)/2) < sz(3) && ...
            (x - szt_c(1)/2) > 0 && (y - szt_c(2)/2) > 0 && (z - szt_c(3)/2) > 0 ) 
            ind_flag = false;
        else
            n2 = n2 + 1;
            if n2 > 10
                coord = false;
                return
            end
        end
    end
    
    xinds = x:x+szt_c(1)-1;
    yinds = y:y+szt_c(2)-1;
    zinds = z:z+szt_c(3)-1;
    
    % Pull out area around the point in the mask image
    tmp_mask = lung_mask(xinds, yinds, zinds);
    
    % Compare tumor with masked area, tumor outside the lungs causes a fail
    agreement = sum(tumor_crop(:) .* tmp_mask(:));
    
    % If the tumor does not go outside the lungs break loop
    if agreement > overlap_thresh
        
        flag = false;
        
    else
        n = n + 1;
        if n > 10
            coord = false;
            return
        end

    end
    
end

%% Create output variables

% Return tumor coordinates
coord = [x, y, z];

% Modify coordinates to account for crops
% coord = coord + starts - tu_starts;

% Remove tumor position from the mask
tmp = lung_mask(x:x+szt(1)-1, y:y+szt(2)-1, z:z+szt(3)-1);
tmp(tumor_mask(:)) = 0;
lung_mask(x:x+szt(1)-1, y:y+szt(2)-1, z:z+szt(3)-1) = tmp;

end

