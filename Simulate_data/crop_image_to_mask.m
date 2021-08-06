function [X_out, lab_out, mask_out] = crop_image_to_mask(X, lab, mask, buffer)
%Crops an input image and mask to the size of the mask + buffer

% Account for trachea
zsums = squeeze(sum(sum(mask,1),2));
ind = find(zsums > 0.04*max(zsums), 1, 'first');

if ind > buffer
     ind = ind - buffer;
end

% Remove trachea
X = X(:,:,ind:end);
lab = lab(:,:,ind:end);
mask = mask(:,:,ind:end);

% Set up variables
X_out = X;
lab_out = lab;
mask_out = mask;
axes = 1:3;

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
    
    % Reorder axes
    mask_out = permute(mask_out, axes);
    X_out = permute(X_out, axes);
    lab_out = permute(lab_out, axes);
    
    % Crop
    mask_out = mask_out(:, :, start:stop);
    X_out = X_out(:, :, start:stop);
    lab_out = lab_out(:, :, start:stop);
    
    % Reverse permute operation
    mask_out = ipermute(mask_out, axes);
    X_out = ipermute(X_out, axes);
    lab_out = ipermute(lab_out, axes);
    
end


end

