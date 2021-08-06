%% Create_lung_masks_210114.m
clear; clc; close all;

%% Set up paths
set(0,'DefaultFigureWindowStyle', 'modal')
base_outpath = '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/PreparedSets/';
outpath_no_contrast = fullfile(base_outpath, 'no_contrast');
outpath_vas_contrast = fullfile(base_outpath, 'vas_contrast');
outpath_other_contrast = fullfile(base_outpath, 'other_contrast');
crop_buffer = 30;

no_contrast = {
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/180216_2',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/180216_3',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/180216_5',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007700',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007701',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007702',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007703'
    };

vas_contrast = {
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007723',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007731',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007733',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007735',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007737',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007739',
    };

other_contrast = {
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007741',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007743',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007747',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007751',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007753',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007755',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007757',...
   '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/Nifti/C007759'
};

if ~exist(outpath_no_contrast, 'dir'); mkdir(outpath_no_contrast); end
if ~exist(outpath_vas_contrast, 'dir'); mkdir(outpath_vas_contrast); end
if ~exist(outpath_other_contrast, 'dir'); mkdir(outpath_other_contrast); end


%% Process - no contrast

outpath = outpath_no_contrast;
base_paths = no_contrast;
for z = 1:length(base_paths)
    base_path = base_paths{z};
    fprintf('Working on %s\n', base_path);
    
    % Find image files
    im_files = dir(fullfile(base_path, '**/*.nii'));
    im_files = fullfile({im_files(:).folder}, {im_files(:).name});
    
    % Get reconstruction and label
    lab_inds = contains(im_files, 'seg');
    im_file = im_files{~lab_inds};
    lab_file = im_files{lab_inds};
    
    % Create output files
    out_tmp = fullfile(outpath, sprintf('%02d', z));
    if ~exist(out_tmp, 'dir'); mkdir(out_tmp); end
    output_labfile = fullfile(out_tmp, 'label.nii.gz');
    output_mskfile = fullfile(out_tmp, 'mask.nii.gz');
    output_imfile = fullfile(out_tmp, 'image.nii');
        
    % Save data sources
    fid = fopen(fullfile(out_tmp, 'sources.txt'), 'w');
    fprintf(fid, '%s\n', base_path);
    fclose(fid);
    
    % Load image
    X = load_nii(im_file);
    dx = 0.063;
    X = X.img;
    sz = size(X);
    nz = size(X, 3);
    
    % Load label
    lab = load_nii(lab_file);
    lab = lab.img;
    
    % Flip labels if they do not overlap
    if mean(X(logical(lab)), [1,2,3]) < -600
        lab = lab(end:-1:1, end:-1:1, :);
    end
    
    % Load problem masks
    if any(z == [4, 7])
        mask_full = load_nii(fullfile(base_path, 'lung_mask.nii.gz'));
        mask_full = mask_full.img;
        mask_full = mask_full(end:-1:1, end:-1:1, :);
    else
        % Compute lung mask
        [X_crop, mask, mask_full] = make_lung_mask(X, crop_buffer);
    end
    
    figure(1)
    imshowpair(X(:, :, floor(nz/2)), mask_full(:, :, floor(nz/2)));
    sname = fullfile(fileparts(out_tmp), sprintf('%02d.png', z));
    print(sname, '-dpng');
    
    % Compute crops
    xind = find(sum(mask_full, [2, 3]));
    xind = xind(:)';
    xind = xind([1, end]) + [-30, 30];
    xind(1) = max(1, xind(1));
    xind(2) = min(sz(1), xind(2));
    
    yind = find(sum(mask_full, [1, 3]));
    yind = yind(:)';
    yind = yind([1, end]) + [-30, 30];
    yind(1) = max(1, yind(1));
    yind(2) = min(sz(2), yind(2));
    
    zind = find(sum(mask_full, [1, 2]));
    zind = zind(:)';
    zind = zind([1, end]) + [-30, 30];
    zind(1) = max(1, zind(1));
    zind(2) = min(sz(3), zind(2));
    
    % Apply crops
    mask_full = mask_full(xind(1):xind(2), :, :);
    mask_full = mask_full(:, yind(1):yind(2), :);
    mask_full = mask_full(:, :, zind(1):zind(2));
    
    X = X(xind(1):xind(2), :, :);
    X = X(:, yind(1):yind(2), :);
    X = X(:, :, zind(1):zind(2));
    
    lab = lab(xind(1):xind(2), :, :);
    lab = lab(:, yind(1):yind(2), :);
    lab = lab(:, :, zind(1):zind(2));
    
    % Save outputs
    save_nii(make_nii(lab, dx), output_labfile);
    save_nii(make_nii(uint16(mask_full), dx), output_mskfile);
    save_nii(make_nii(single(X), dx), output_imfile);

    fprintf('\n');
end


fprintf('\n\nDone!\n\n');


%% Process - vascular contrast

outpath = outpath_vas_contrast;
base_paths = vas_contrast;
for z = 1:length(base_paths)
    base_path = base_paths{z};
    fprintf('Working on %s\n', base_path);
    
    % Find image files
    im_files = dir(fullfile(base_path, '**/*.nii'));
    im_files = fullfile({im_files(:).folder}, {im_files(:).name});
    
    % Get reconstruction and label
    lab_inds = contains(im_files, 'seg');
    im_file = im_files{~lab_inds};
    lab_file = im_files{lab_inds};
    
    % Create output files
    out_tmp = fullfile(outpath, sprintf('%02d', z));
    if ~exist(out_tmp, 'dir'); mkdir(out_tmp); end
    output_labfile = fullfile(out_tmp, 'label.nii.gz');
    output_mskfile = fullfile(out_tmp, 'mask.nii.gz');
    output_imfile = fullfile(out_tmp, 'image.nii');
    
    % Save data sources
    fid = fopen(fullfile(out_tmp, 'sources.txt'), 'w');
    fprintf(fid, '%s\n', base_path);
    fclose(fid);
    
    % Load image
    X = load_nii(im_file);
    dx = 0.063;
    X = X.img;
    sz = size(X);
    nz = size(X, 3);
    
    % Load label
    lab = load_nii(lab_file);
    lab = lab.img;
    
    % Flip labels if they do not overlap
    if mean(X(logical(lab)), [1,2,3]) < -600
        lab = lab(end:-1:1, end:-1:1, :);
    end

    % Load problem masks
    if any(z == [4])
        mask_full = load_nii(fullfile(base_path, 'lung_mask.nii.gz'));
        mask_full = mask_full.img;
        mask_full = mask_full(end:-1:1, end:-1:1, :);
    else
        % Compute lung mask
        [X_crop, mask, mask_full] = make_lung_mask(X, crop_buffer);
    end

    figure(1)
    imshowpair(X(:, :, floor(nz/2)), mask_full(:, :, floor(nz/2)));
    sname = fullfile(fileparts(out_tmp), sprintf('%02d.png', z));
    print(sname, '-dpng');
    
    % Compute crops
    xind = find(sum(mask_full, [2, 3]));
    xind = xind(:)';
    xind = xind([1, end]) + [-30, 30];
    xind(1) = max(1, xind(1));
    xind(2) = min(sz(1), xind(2));
    
    yind = find(sum(mask_full, [1, 3]));
    yind = yind(:)';
    yind = yind([1, end]) + [-30, 30];
    yind(1) = max(1, yind(1));
    yind(2) = min(sz(2), yind(2));
    
    zind = find(sum(mask_full, [1, 2]));
    zind = zind(:)';
    zind = zind([1, end]) + [-30, 30];
    zind(1) = max(1, zind(1));
    zind(2) = min(sz(3), zind(2));
    
    % Apply crops
    mask_full = mask_full(xind(1):xind(2), :, :);
    mask_full = mask_full(:, yind(1):yind(2), :);
    mask_full = mask_full(:, :, zind(1):zind(2));
    
    X = X(xind(1):xind(2), :, :);
    X = X(:, yind(1):yind(2), :);
    X = X(:, :, zind(1):zind(2));
    
    lab = lab(xind(1):xind(2), :, :);
    lab = lab(:, yind(1):yind(2), :);
    lab = lab(:, :, zind(1):zind(2));
    
    % Save outputs
    save_nii(make_nii(lab, dx), output_labfile);
    save_nii(make_nii(uint16(mask_full), dx), output_mskfile);
    save_nii(make_nii(single(X), dx), output_imfile);
    
    fprintf('\n');
    
end


fprintf('\n\nDone!\n\n');



%% Process - other contrast

outpath = outpath_other_contrast;
base_paths = other_contrast;
for z = 1:length(base_paths)
    base_path = base_paths{z};
    fprintf('Working on %s\n', base_path);
    
    % Find image files
    im_files = dir(fullfile(base_path, '**/*.nii'));
    im_files = fullfile({im_files(:).folder}, {im_files(:).name});
    
    % Get reconstruction and label
    lab_inds = contains(im_files, 'seg');
    im_file = im_files{~lab_inds};
    lab_file = im_files{lab_inds};
    
    % Create output files
    out_tmp = fullfile(outpath, sprintf('%02d', z));
    if ~exist(out_tmp, 'dir'); mkdir(out_tmp); end
    output_labfile = fullfile(out_tmp, 'label.nii.gz');
    output_mskfile = fullfile(out_tmp, 'mask.nii.gz');
    output_imfile = fullfile(out_tmp, 'image.nii');
        
    % Save data sources
    fid = fopen(fullfile(out_tmp, 'sources.txt'), 'w');
    fprintf(fid, '%s\n', base_path);
    fclose(fid);
    
    % Load image
    X = load_nii(im_file);
    dx = 0.063;
    X = X.img;
    sz = size(X);
    nz = size(X, 3);
    
    % Load label
    lab = load_nii(lab_file);
    lab = lab.img;
    
    % Flip labels if they do not overlap
    if mean(X(logical(lab)), [1,2,3]) < -600
        lab = lab(end:-1:1, end:-1:1, :);
    end

    % Load problem masks
    if any(z == [1, 2, 3])
        mask_full = load_nii(fullfile(base_path, 'lung_mask.nii.gz'));
        mask_full = mask_full.img;
        mask_full = mask_full(end:-1:1, end:-1:1, :);
    else
        % Compute lung mask
        [X_crop, mask, mask_full] = make_lung_mask(X, crop_buffer);
    end

    figure(1)
    imshowpair(X(:, :, floor(nz/2)), mask_full(:, :, floor(nz/2)));
    sname = fullfile(fileparts(out_tmp), sprintf('%02d.png', z));
    print(sname, '-dpng');
    
    % Compute crops
    xind = find(sum(mask_full, [2, 3]));
    xind = xind(:)';
    xind = xind([1, end]) + [-30, 30];
    xind(1) = max(1, xind(1));
    xind(2) = min(sz(1), xind(2));
    
    yind = find(sum(mask_full, [1, 3]));
    yind = yind(:)';
    yind = yind([1, end]) + [-30, 30];
    yind(1) = max(1, yind(1));
    yind(2) = min(sz(2), yind(2));
    
    zind = find(sum(mask_full, [1, 2]));
    zind = zind(:)';
    zind = zind([1, end]) + [-30, 30];
    zind(1) = max(1, zind(1));
    zind(2) = min(sz(3), zind(2));
    
    % Apply crops
    mask_full = mask_full(xind(1):xind(2), :, :);
    mask_full = mask_full(:, yind(1):yind(2), :);
    mask_full = mask_full(:, :, zind(1):zind(2));
    
    X = X(xind(1):xind(2), :, :);
    X = X(:, yind(1):yind(2), :);
    X = X(:, :, zind(1):zind(2));
    
    lab = lab(xind(1):xind(2), :, :);
    lab = lab(:, yind(1):yind(2), :);
    lab = lab(:, :, zind(1):zind(2));
    
    % Save outputs
    save_nii(make_nii(uint8(lab), dx), output_labfile);
    save_nii(make_nii(uint8(mask_full), dx), output_mskfile);
    save_nii(make_nii(single(X), dx), output_imfile);
    
    fprintf('\n');

    
end


fprintf('\n\nDone!\n\n');