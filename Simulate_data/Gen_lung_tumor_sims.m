%% Gen_lung_tumor_sims_200916.m

clear; clc; close all;

%% Make tumor masks
% base_path = 'E:\\CT Data\\LungTumors\\SimTumors\\181207\\';
%base_path = '/media/justify/data/SimTumors/200916';
base_path = '/home/matt/Documents/Working_data/200916';
outpath_im = fullfile(base_path, 'training', 'imagesTr');
outpath_lb = fullfile(base_path, 'training', 'labelsTr');
outpath_im_ts = fullfile(base_path, 'training', 'imagesTs');
outpath_lb_ts = fullfile(base_path, 'training', 'labelsTs');
outpath = fullfile(base_path, 'working');

if ~exist(outpath_im, 'dir'); mkdir(outpath_im); end
if ~exist(outpath_lb, 'dir'); mkdir(outpath_lb); end
if ~exist(outpath_im_ts, 'dir'); mkdir(outpath_im_ts); end
if ~exist(outpath_lb_ts, 'dir'); mkdir(outpath_lb_ts); end
if ~exist(outpath, 'dir'); mkdir(outpath); end

% Load file with image and mask locations
data_file = 'Animals_tumors_segmented_09-10-2020_1.xlsx';
dat = readcell(data_file, 'range', 2);
im_files = dat(11:end,2);
num_ims = length(im_files);

% Convert to OS fileseparator
im_files = replace(im_files, '\', filesep);

% Remove drive
new_drive = '/media/justify';
if ~ispc
    for z = 1:num_ims 
        ind = strfind(im_files{z}, ':');
        if ~isempty(ind)
            inds = 1:ind;
            tmp = im_files{z}(1:ind);
            im_files{z} = replace(im_files{z}, tmp, new_drive);
            im_files{z} = fullfile(fileparts(im_files{z}), 'X1_BF.nii');
        end
    end
end

% Get lung segmentation image filename
seg_files = cell(1, num_ims);
for z = 1:num_ims 
    %seg_files{z} = fullfile(fileparts(im_files{z}), 'mask_cropped.nii.gz');
    seg_files{z} = fullfile(fileparts(im_files{z}), 'mask.nii.gz');
end


%% Load image and mask
fprintf('Loading clean CT images\n');

Xs = cell(1, num_ims );
masks = cell(1, num_ims );
wattn = zeros(1, num_ims);

% Closing element
r = 3;
SE = strel('sphere',r);

% Crop main volumes
buffer = 80;

for z = 1:num_ims 
    fprintf('\t%s\n', im_files{z});
    
    % Load image
    im_file = im_files{z};
    im = load_nii(im_file);
    im = im.img;
    
    % Load mask
    im_file = seg_files{z};
    mask_close = load_nii(im_file);
    mask_close = mask_close.img;
    
    % Crop
    [im, ~, mask_close] = crop_image_to_mask(im, zeros(size(im)), mask_close, buffer);
    
    % Load water attnenuation
    fname = fullfile(fileparts(im_files{z}), 'recon_params.json');
    RP = loadjson(fname);
    wattn(z) = RP.mu1;
    
    % Convert images from HU to linear attenuation
    im = ((im / 1000) + 1) * wattn(z);
    
    % Close mask
    mask_close = imclose(mask_close, SE);
    
    % Pad image sizes to multiples of 8
    sz = size(im);
    nsz = 8*ceil(sz/8);
    nim = zeros(nsz);
    nim(1:sz(1), 1:sz(2), 1:sz(3)) = im;
    im = nim;
    nmk = zeros(nsz);
    nmk(1:sz(1), 1:sz(2), 1:sz(3)) = mask_close;
    mask_close = nmk;
    
    % Place in X
    Xs{z} = im;
    masks{z} = mask_close;
end

% Attenuation of water
attn_wat = mean(wattn);

%% Examine images and segs
%{
for z = 1:num_ims
    
    xtmp = X{z};
    mtmp = mask{z};
    fprintf('%d:\n\tSize X:\t\t(%d, %d, %d)\n\tSize mask:\t(%d, %d, %d)\n', z, size(xtmp), size(mtmp));
    
    s = floor(size(xtmp)/2);
    
    figure()
    imshowpair(xtmp(:, :, s(3)), mtmp(:, :, s(3)))
end

fprintf('\n')
%}

%% Load tumors
fprintf('Loading tumors\n');
mask_files = {'Tumor A Segmentation.nii',...
              'Tumor B Segmentation.nii',...
              'Tumor C Segmentation.nii'};
mask_files = fullfile(base_path, mask_files);
          
% Load tumors from file
tumor_files = dat(2:7, 2);
tseg_files = dat(2:7, 3);
num_segs = length(tumor_files);

% Convert to OS fileseparator
tumor_files = replace(tumor_files, '\', filesep);
tseg_files = replace(tseg_files, '\', filesep);

% Remove drive
new_drive = '/media/blkbeauty5';
if ~ispc
    for z = 1:num_segs
        ind = strfind(tumor_files{z}, ':');
        if ~isempty(ind)
            inds = 1:ind;
            tmp = tumor_files{z}(1:ind);
            tumor_files{z} = replace(tumor_files{z}, tmp, new_drive);
            tseg_files{z} = replace(tseg_files{z}, tmp, new_drive);
        end
    end
end

% Filter out bad segmentations
bad = {'523035', '408468'};
inds = contains(tumor_files, bad);
tumor_files = tumor_files(~inds);
tseg_files = tseg_files(~inds);

% Remove white space
tumor_files = strip(tumor_files);
tseg_files = strip(tseg_files);

num_segs = length(tumor_files);
tumor_masks = cell(1, num_segs);
tumor_ims = cell(1, num_segs);
tumor_mns = zeros(1, num_segs);
tumor_stds = zeros(1, num_segs);

for z = 1:num_segs 
    
    % Load image
    im_file = tumor_files{z};
    im = load_nii(im_file);
    im = im.img;
    
    % Load mask
    im_file = tseg_files{z};
    tumor_mask = load_nii(im_file);
    tumor_mask = logical(tumor_mask.img);

    [~, nm] = fileparts(fileparts(im_file));
    fprintf('Scan: %s\n', nm)
    
    % Convert images from HU to linear attenuation
    im = ((im / 1000) + 1) * wattn(z);
    
    % Get segmentation volume properties
    r = regionprops(tumor_mask, 'Area', 'BoundingBox', 'PixelIdxList');
    
    % Select largest area
    [~, mx_areas]  = max([r(:).Area]);
    rm = r(mx_areas);
    
    % Mask image
    msk = false(size(im));
    msk(rm.PixelIdxList) = 1;
    im = im .* msk;   
    
    % Create cropping indices
    inds = floor(rm.BoundingBox);
    
    % Pad to multiple of 2
    minds = cell(1, 3);
    for j = 1:3
        ln = length(inds(j):inds(j)+inds(j+3));
        pad = 2*ceil(ln/2) - ln;
        minds{j} = inds(j):inds(j)+inds(j+3)+pad; 
    end
    
    % Crop to mask
    msk = msk(minds{2}, minds{1}, minds{3});
    im = im(minds{2}, minds{1}, minds{3});
    
    % Collect final images
    tumor_masks{z} = msk;
    tumor_ims{z} = im;
    
    % Get image properties
    tumor_mns(z) = mean(im(msk(:)));
    tumor_stds(z) = std(im(msk(:)));

end

%% Show tumors

%{
for z = 1:num_segs
    % Load image
    im_file = tumor_files{z};
    im = load_nii(im_file);
    im = im.img;
    
    % Load mask
    im_file = tseg_files{z};
    tmp_mask = load_nii(im_file);
    tmp_mask = logical(tmp_mask.img);
    
    [~, nm] = fileparts(fileparts(im_file));
    fprintf('Scan: %s\n', nm)
    
    % Get segmentation volume properties
    r = regionprops(tmp_mask, 'Area', 'BoundingBox', 'PixelIdxList');
    fprintf('\tNumber of regions: %d\n', length(r))
    for zz = 1:length(r)
        fprintf('\t\t%d\n', r(zz).Area);
    end
    
    % Select largest area
    [~, mx_areas]  = max([r(:).Area]);
    rm = r(mx_areas);
    
    % Mask image
    msk = false(size(im));
    msk(rm.PixelIdxList) = 1;
    im = im .* msk;   
    
    % Create cropping indices
    inds = floor(rm.BoundingBox);
%     inds(1:3) = inds(1:3) - 1;
%     inds(4:6) = inds(4:6) + 1;
    
    % Crop to mask
    msk = msk(inds(2):inds(2)+inds(5), inds(1):inds(1)+inds(4), inds(3):inds(3)+inds(6));
    im = im(inds(2):inds(2)+inds(5), inds(1):inds(1)+inds(4), inds(3):inds(3)+inds(6));
    
    figure()
    imshowpair(im(:, :, floor(inds(6)/2)), msk(:, :, floor(inds(6)/2)), 'montage')
    [~, nm] = fileparts(fileparts(im_file));
    title(sprintf('Scan: %s', nm));
    
end

%}

%% Generate sets
start_time = tic;
num_vols = 10;
num_tumors = 2 * ones(num_vols, 1);
vox_size = 0.063^3;
test_size = 0.2; % pct split

% Random changes
shear_limits = [0.7, 1.3];  
mag_limits = [0.9, 1.1];

% Blue parameters
sigma = 3.0;

% Seed the random number generator
rng(1);
seeds = randi(1000, 1000, 1);
seed_counter = 1;

X = {};
Y = {};
move_fact = 8;
szX = size(Xs{1});

for j = 1:num_vols
    tic;
    fprintf('Generating set %d\n', j);
    
    % Pick random control mousse
    scan_ind = mod(j-1, num_ims) + 1;  %ceil(num_ims*rand);
    fprintf('\tScan index: %d\n', scan_ind);
    X{1} = Xs{scan_ind};
    
    lung_mask = masks{scan_ind};
    lab = zeros(size(X{1}), 'logical');
    
    % Futher close mask to reduce effect of airways
    r = 6;
    SE = strel('sphere',r);
    mask_air_closed = lung_mask; %imclose(lung_mask, SE);
    
    % Set up outout file
    csv_file = sprintf('%slabels_%d.csv', outpath, j);
    f = fopen(csv_file, 'w');
    fprintf(f, 'tumor index,x,y,z,volume (voxels),volume (mm^3)\n');
    
    fprintf('\tAdding tumor:    ');
    for t = 1:num_tumors(j)
        fprintf('\b\b\b%02d\n', t);
        tum_flag = true;
        num_tries = 1;
        while tum_flag
            % Pic a random tumor
            tumor_ind = ceil(3*rand);
            tumor_mask = tumor_masks{tumor_ind};
            tumor = tumor_ims{tumor_ind};
            
            % Seed the random number generator for tumor warps
            seed = seeds(seed_counter);
            seed_counter = seed_counter + 1;

            % Warp the tumor mask
            %tmp_mask = warp_tumors(tmp_mask, shear_limits);
            rng(seed);
            tumor = warp_tumors(tumor, shear_limits);
            rng(seed);
            tumor_mask = warp_tumors(tumor_mask, shear_limits);

            % Add texture to the tumor
            %tmp_mn = tumor_mn(tumor_ind);
            %tmp_std = tumor_std(tumor_ind);
            %tumor = tmp_std*randn(size(tmp_mask)) + tmp_mn;

            % Select tumor location
            [coord, tmp_lung_mask2] = loc_in_mask( lung_mask, tumor_mask, mask_air_closed, 0.95 );

            if coord
                tum_flag = false;
                num_tries = num_tries + 1;
                fprintf('Failed to find tumor location, retrying with new tumor.\n\t\t   ');
                if num_tries > 5
                    tum_flag = true;
                end
            end
            
        end
        
        szt = size(tumor);
        xinds = coord(1):coord(1)+szt(1)-1;
        yinds = coord(2):coord(2)+szt(2)-1;
        zinds = coord(3):coord(3)+szt(3)-1;
        
        ccoord = round(coord + szt/2);
        
        % Redefine if out of range
        ccoord = min(ccoord, size(X{1}));

        % Insert tumor into volume
        nxinds = xinds; %round(xinds + move_fact*dfield{i}(ccoord(1), ccoord(2), ccoord(3), 1));
        nyinds = yinds; %round(yinds + move_fact*dfield{i}(ccoord(1), ccoord(2), ccoord(3), 2));
        nzinds = zinds; %round(zinds + move_fact*dfield{i}(ccoord(1), ccoord(2), ccoord(3), 3));

        % Create volume with tumor mask
        ins_mask = zeros(size(X{1}), 'logical');
        ins_mask(nxinds, nyinds, nzinds) = tumor_mask;
        ins_mask_dia = imgaussfilt3(single(ins_mask), sigma);

        % Apply lung mask to tumor mask
        ins_mask_dia = ins_mask_dia .* lung_mask;
        ins_mask_dia2 = ins_mask_dia .* ~ins_mask;

        % Normalize outer mask
        ins_mask_dia2 = ins_mask_dia2 / max(ins_mask_dia2(:));

        % Create volume with tumor placed in image
        ins_tumor = X{1};
        ins_tumor(ins_mask) = tumor(tumor_mask);
        ins_tumor_dia = imgaussfilt3(single(ins_tumor), sigma);

        % Place tumor inside image
        X{1}(ins_mask) = ins_tumor(ins_mask);

        % Blend tumor edges
        tmp = X{1} .* (1 - ins_mask_dia2) + ins_tumor_dia .* ins_mask_dia2;
        X{1} = max(X{1}, tmp);

        % Add tumor to label
        lab(ins_mask(:)) = 1;
        
        % Update lung mask
        lung_mask = tmp_lung_mask2;
        
        % Write CSV file with tumor locs and volumes
        tvol = sum(tumor_mask(:));
        fprintf(f, '%d,%d,%d,%d,%d,%f\n', t, ccoord(1), ccoord(2), ccoord(3), tvol, tvol*vox_size);
        
    end
    fclose(f);        
    
    % Forward project volumes
    fprintf('\tForward projecting\n');
    Y = genprojs_synth_lung_tumors( X{1}, outpath );
    
    % Simulate respiratory gating
        
    %%%%% Gated %%%%%
    Y_tmp = Y;

    % Perform reconstruction
    fprintf('\tBack projecting\n');
    X1 = single_recon_synth_lung_tumors( size(X{1}), Y_tmp, outpath );

    % Center the image on the lung mask
    mask_cent = masks{scan_ind};

    % Pick random magnification
    mag = range(mag_limits) * rand() + min(mag_limits);

    % Magnify images and mask
    X1_mg = imresize3(X1, mag, 'OutputSize', size(X1));
    lab_mg = imresize3(single(lab), mag, 'OutputSize', size(X1));
    mask_mg = imresize3(single(mask_cent), mag, 'OutputSize', size(X1));

    % Convert masks and label to logical
    lab_mg = lab_mg > 0.5;
    % mask_mg = mask_mg > 0.5;

    % Crop labels and recons to around the mask
    buffer = 30;
    [X1, clab, lmask_cropped] = crop_image_to_mask(X1_mg, lab_mg, mask_mg, buffer);

    % Convert to HU
    X1 = 1000 * (X1/attn_wat - 1);
    fprintf('\tSize of X: %d, %d, %d\n', size(X1, 1), size(X1, 2), size(X1, 3));

    if j < length(num_tumors) * (1 - test_size)

        % Save label
        fprintf('\tSaving images\n');    
        savelab = sprintf('%s/lung_%02d.nii', outpath_lb, j);
        save_lab{j} = savelab;
        save_nii(make_nii(single(clab)), savelab)

        % Save image
        savefile = sprintf('%s/lung_%02d.nii', outpath_im, j);
        save_files{j} = savefile;
        save_nii(make_nii(single(X1)), savefile)

    else


        % Save label
        fprintf('\tSaving images\n');    
        savelab = sprintf('%s/lung_%02d.nii', outpath_lb_ts, j);
        save_lab{j} = savelab;
        save_nii(make_nii(single(clab)), savelab)

        % Save image
        savefile = sprintf('%s/lung_%02d.nii', outpath_im_ts, j);
        save_files{j} = savefile;
        save_nii(make_nii(single(X1)), savefile)

    end

    fprintf('\t\tDone in %0.2f seconds\n', toc);

end

% Get reconstruction time
time = toc(start_time);
thours = floor(time/3600);
nsecs = (time - thours*3600);
tmins = floor(nsecs / 60);
tsecs = floor(nsecs - tmins * 60);

fprintf('\nAll finished with %d sets.\n\tProcessing took:\t', num_vols);
fprintf('%d hours, %d minutes, %d seconds\n\n', thours, tmins, tsecs);
