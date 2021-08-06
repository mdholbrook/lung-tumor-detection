%% Sort_training_data_120304.m
clear; clc;

outpath_train = '/home/matt/Documents/Working_data/LungTumors/Real_xfer_210305/train';
outpath_test = '/home/matt/Documents/Working_data/LungTumors/Real_xfer_210305/test';

% Clear outpaths
mkdir(outpath_train);
mkdir(outpath_test);

%% Old sets

train_path = '/home/matt/Documents/Projects/vnet_lung_tumors/data/training_real';
test_path = '/home/matt/Documents/Projects/vnet_lung_tumors/data/testing_real';

% Get starting output folder
ntrain = 1;
ntest = 1;

path = train_path;
[~,im_type,~] = fileparts(path);

% Get filenames
fnames = dir(path);
fnames = fnames([fnames.isdir]);
fnames = sort({fnames(:).name});
fnames = fnames(3:end);

fnames = fullfile(path, fnames, '*');

for zz = 1:length(fnames)

    out_folder = fullfile(outpath_train, sprintf('%02d', ntrain));
    if ~exist(out_folder, 'dir'), mkdir(out_folder); end
    copyfile(fnames{zz}, out_folder);
    ntrain = ntrain + 1;

    % Label the image source
    fid = fopen(fullfile(out_folder, 'image_type.txt'), 'w');
    fprintf(fid, '%s', im_type);
    fclose(fid);
end


% Test files
path = test_path;
im_type = 'old real data';

% Get filenames
fnames = dir(path);
fnames = fnames([fnames.isdir]);
fnames = sort({fnames(:).name});
fnames = fnames(3:end);
fnames = fnames(~contains(fnames, 'Analysis'));


fnames = fullfile(path, fnames, '*');

for zz = 1:length(fnames)

    out_folder = fullfile(outpath_test, sprintf('%02d', ntest));
    if ~exist(out_folder, 'dir'), mkdir(out_folder); end
    copyfile(fnames{zz}, out_folder);
    ntest = ntest + 1;

    % Label the image source
    fid = fopen(fullfile(out_folder, 'image_type.txt'), 'w');
    fprintf(fid, '%s', im_type);
    fclose(fid);
end



fprintf('Done with tumor sets\n');


%% Set up paths - new sets

% outpath_train = '/home/matt/Documents/Working_data/LungTumors/Real_xfer_210304/train';
% outpath_test = '/home/matt/Documents/Working_data/LungTumors/Real_xfer_210304/test';
test_split = 0.3;

paths = {
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/PreparedSets/no_contrast',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/PreparedSets/other_contrast/',...
    '/home/matt/Documents/Working_data/LungTumors/Tumors_210216/PreparedSets/vas_contrast',...
    };


for z = 1:length(paths)
    path = paths{z};
    [~,im_type,~] = fileparts(path);
    
    % Get filenames
    fnames = dir(path);
    fnames = fnames([fnames.isdir]);
    fnames = sort({fnames(:).name});
    fnames = fnames(3:end);
    
    num_sets = length(fnames);
    num_train = round(num_sets  * (1-test_split));
    
    fnames = fullfile(path, fnames, '*');
    
    for zz = 1:length(fnames)
        
        if zz <= num_train
            out_folder = fullfile(outpath_train, sprintf('%02d', ntrain));
            if ~exist(out_folder, 'dir'), mkdir(out_folder); end
            copyfile(fnames{zz}, out_folder);
            ntrain = ntrain + 1;
        else
            out_folder = fullfile(outpath_test, sprintf('%02d', ntest));
            if ~exist(out_folder, 'dir'), mkdir(out_folder); end
            copyfile(fnames{zz}, out_folder);
            ntest = ntest + 1;
        end
        
        % Label the image source
        fid = fopen(fullfile(out_folder, 'image_type.txt'), 'w');
        fprintf(fid, '%s', im_type);
        fclose(fid);
    end
    
end

fprintf('Done with tumor sets\n');


%% Add clean lungs

path = '/home/matt/Documents/Working_data/LungTumors/CleanTumors/';
im_type = 'clean';

% Get filenames
fnames = dir(path);
fnames = fnames([fnames.isdir]);
fnames = sort({fnames(:).name});
fnames = fnames(3:end);

num_sets = length(fnames);
num_train = round(num_sets  * (1-test_split));

fnames = fullfile(path, fnames, '*');

for zz = 1:length(fnames)

    if zz <= num_train
        out_folder = fullfile(outpath_train, sprintf('%02d', ntrain));
        if ~exist(out_folder, 'dir'), mkdir(out_folder); end
        copyfile(fnames{zz}, out_folder);
        ntrain = ntrain + 1;
    else
        out_folder = fullfile(outpath_test, sprintf('%02d', ntest));
        if ~exist(out_folder, 'dir'), mkdir(out_folder); end
        copyfile(fnames{zz}, out_folder);
        ntest = ntest + 1;
    end

    % Label the image source
    fid = fopen(fullfile(out_folder, 'image_type.txt'), 'w');
    fprintf(fid, '%s', im_type);
    fclose(fid);
end

fprintf('Done with clean sets\n');