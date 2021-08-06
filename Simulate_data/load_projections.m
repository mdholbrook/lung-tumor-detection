function Y1 = load_projections(y_file, CTX)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
nt = tic;
if exist(y_file, 'file')
    fprintf('\tLoading preprocessed projections:\n\t\t%s\n', y_file);
    Y1 = load_nii(y_file);
    Y1 = Y1.img;
else
    
    % If it does not exist find the raw projections
    fraw = y_file;
    for z = 1:3
        fraw = fileparts(fraw);
    end
    fraw = fullfile(fraw, 'vhr1', 'set1');
    nfile = fullfile(fraw, dir(fullfile(fraw, 'set1*.nii')).name);
    
    % Get number of projections
    info = load_untouch_header_only(nfile);
    np = info.dime.dim(4);
    
    % Convert filename to cell array
    nfile = {nfile};
    
    % Get air and dark files
    airname = fullfile(fileparts(y_file), 'airraw.nii.gz');
    darkname = fullfile(fileparts(y_file), 'dark.nii.gz');
    defectmap = fullfile(fileparts(y_file), 'defect.nii.gz');
    
    % Set up path and file names
    variable_exposure_correction = CTX.govex;
    outpath = fileparts(y_file);
    
    % Determine projection format
    sets = CTX.set;
    chain = 1;
    CTX.chain1 = 1;

    % Get crop information
    xposition = CTX.xposition;

    xcrop = [xposition(1), sum(xposition([1 3]))-1];
    zcrop = [xposition(2), sum(xposition([2 4]))-1];

    % New projection sizes
    nu_c = xposition(3);
    nv_c = xposition(4);

    % Read in airraw file information
    [~, ~, ext] = fileparts(airname);

    if isempty(ext)  % Look in nested folder
        tmp = dir(fullfile(airname, 'vhr1', 'set1', 'set1*.nii'));
        tmp = fullfile(airname, 'vhr1', 'set1', tmp(1).name);
        [~, ~, ext] = fileparts(tmp);
        if isempty(ext)
            error('Airraw not found:\n\t%s', airname);
        else
            airname = tmp;
        end
    end

    if strncmp(ext, '.tif', 10)
        info = imfinfo(airname);
        nadp = numel(info);

        air = zeros(nu_c,nv_c,nadp);
        air_save = zeros(info(1).Width, info(1).Height, nadp);

        for i = 1:nadp
            air_ = imread(airname,i)';
            air_save(:,:,i) = air_;
            air(:,:,i) = air_(xcrop(1):xcrop(2),zcrop(1):zcrop(2));
        end
    elseif contains('.nii.gz', ext, 'IgnoreCase', true) 
        air_ = load_nii(airname);
        air_ = air_.img;
        air_save = air_;
        air = air_(xcrop(1):xcrop(2),zcrop(1):zcrop(2), :);
    end

    clearvars air_

    % Read in dark file information
    [~, ~, ext] = fileparts(darkname);

    if strncmp(ext, '.tif', 10)
        info = imfinfo(darkname);
        nadp = numel(info);

        dark = zeros(nu_c,nv_c,nadp);
        dark_save = zeros(info(1).Width, info(1).Height, nadp);

        for i = 1:nadp
            dark_ = imread(darkname,i)';
            dark_save(:,:,i) = dark_;
            dark(:,:,i) = dark_(xcrop(1):xcrop(2),zcrop(1):zcrop(2));           
        end
    elseif contains('.nii.gz', ext, 'IgnoreCase', true)
        dark_ = load_nii(darkname);
        dark_ = dark_.img;
        dark_save = dark_;
        dark = dark_(xcrop(1):xcrop(2),zcrop(1):zcrop(2), :);
    end

    clearvars dark_

    % Read in the defect map
    [~, ~, ext] = fileparts(defectmap);
    if isempty(defectmap)

        % No defect map specified
        defect_save = zeros(size(dark_save,1), size(dark_save,2));
        defect = zeros(size(dark,1), size(dark,2));

    else

        if strncmp(ext, '.tif', 10)

            % Load the defect map
            defect = imread(defectmap)';

            % Flip defect map horizontally and vertically
            defect = defect(:,end:-1:1);
            defect = defect(end:-1:1,:);

        elseif contains('.nii.gz', ext, 'IgnoreCase', true)

            % Load the defect map
            defect = load_nii(defectmap);
            defect = defect.img;

        end

        % Crop defect map
        defect = defect(xcrop(1):xcrop(2),zcrop(1):zcrop(2));

    end


    % Correct projections
    if numel(size(air)) == 2
    elseif size(air,3) > 6
        air = mean(air(:,:,6:end),3);
    else
        air = mean(air,3);
    end
    dark = mean(dark,3);


    % Set defective pixels to 0
    air_dark = air - dark;        
    idx = defect ~= 0;
    air_dark(idx) = 0;   

    % % Convolve over defective pixels
    sig = 1 / (2 * sqrt( 2* log(2) ) );
    G = exp(-(-3:3).^2 ./ ( 2 * sig^2 ));
    G = G./sum(G);

    air_dark_G = convn(convn(air_dark,G,'same'),G','same');

    defect_norm = single(defect == 0);
    defect_norm = convn(convn(defect_norm,G,'same'),G','same');

    air_dark2 = air_dark;

    air_dark2(idx) = air_dark_G(idx)./defect_norm(idx);

    air_dark = air_dark2;

    clearvars air_dark2


    % Read in projections for Y_sums, variable exposure correction
    [~, ~, proj_ext] = fileparts(nfile{1});
    if variable_exposure_correction
        if exist(fullfile(outpath, ['NormalizationData' num2str(chain) '.mat']), 'file')
            load(fullfile(outpath, ['NormalizationData' num2str(chain) '.mat']), 'Y_sums')
        else
            Y_sums = [];
        end
        if numel(Y_sums) ~= np

            fprintf('\tPreparing variable exposure correction...\n')
            Y_sums = zeros(np,1);

            if contains(proj_ext, '.tif','IgnoreCase',true)
                for i = 1:np

                    temp_ = imread(nfile, i)';

                    Y_sums(i) = sum(temp_(:));


                end
        elseif contains('.nii.gz', proj_ext, 'IgnoreCase', true)
                file = [fdir(1:end-5), nfile{1}];
                temp_ = load_nii(file);
                temp_ = temp_.img;
                Y_sums = squeeze(sum(sum(temp_,1),2));
            end

            % Save Y_sums so it does not have to be computed each time the
            % roi changes
            save([outpath 'NormalizationData' num2str(chain) '.mat'],'Y_sums','-mat')
        end
    end


           % % Read in and normalize projections
            n = 1;  % counting variable for loading projections
            c = 0; % counting variable for files written
            seq = 1; % file number
            pr = np;    % projections remaining to be read

            % Set up output array
            Y_out = zeros(8*ceil(nu_c/8), 8*ceil(nv_c/8), np, 'single');

            if contains(char(java.net.InetAddress.getLocalHost.getHostName), 'secretariat')
                mnp = np;
            else
                mnp = 360;  % max number of projections to read at one time
            end
            str = {};

            while pr > 0

                temp = zeros(nu_c,nv_c,mnp);

                if (pr-mnp) < 0
                    mnp = pr;
                    temp = zeros(nu_c,nv_c,mnp);
                end

                pr = pr - mnp;  % number of projections remaining

                fprintf('\tNormalizing projections %0.0f through %0.0f...\n',np-pr-mnp+1,np-pr)

                if contains(proj_ext, '.tif','IgnoreCase',true)
                    for z = 1:mnp

                        % Update filename
                        if n == 1
                            file = [fdir(1:end-5), nfile{seq}];
                        end

                        % Read in projections
                        temp_ = imread(file,n)';
                        temp(:,:,z) = temp_(xcrop(1):xcrop(2),zcrop(1):zcrop(2));

                        n = n + 1;

                        % Update counting variables for each new file
                        if mod(n,nprojset(seq) + 1) == 0

                            seq = seq + 1;
                            n = 1;    

                        end

                    end
                elseif contains('.nii.gz', proj_ext, 'IgnoreCase', true)
                    if ~exist('temp_,','var')
                        file = [nfile{1}];
                        temp_ = load_nii(file);
                        temp_ = temp_.img;

                    end
                    temp = double(temp_(xcrop(1):xcrop(2),zcrop(1):zcrop(2),(np-pr-mnp+1):(np-pr)));     
                end

                Y1 = temp;

                clearvars temp

                Y_dark = bsxfun(@minus,Y1,dark);
                Y_dark(Y_dark < 0) = 0;
                Y_dark = bsxfun(@min,Y_dark,air_dark);

                for z = 1:mnp

                    Y_dark2 = Y_dark(:,:,z);

                    Y_dark2(idx) = 0;

                    Y_dark_G = convn(convn(Y_dark2,G,'same'),G','same');

                    Y_dark2(idx) = Y_dark_G(idx)./defect_norm(idx);

                    Y_dark(:,:,z) = Y_dark2;

                end

                % Beam hardening correction
                if isfield(CTX, 'beamhd')

                    if CTX.beamhd

                        % % Normalize and log transform projections
                        Y1 = real(-log(bsxfun(@rdivide,Y_dark,air_dark)));


                        % Beam hardening correction
                        % coefs_file = 'C:\Users\xray\Documents\MATLAB\ReconGUI_Dexela\BH_Coefficients.txt';

                        if chain == 1     
                            poly = [-0.0334107323329121 0.134625487166717 0.974957979722303 0.0015061451777908];%50       
                        elseif chain == 2
                            poly = [0.0404298053932853 0.0250841764469297 1.05279596809865 -0.00898936957815281];%40       
                        end

                        Y1 = polyval(poly, Y1);

                        Y1 = exp(-Y1).*air_dark;
                        Y_dark = Y1;

                    end   

                end


                % % Variable exposure correction
                if variable_exposure_correction
                    Y1 = bsxfun(@times,Y_dark,reshape(bsxfun(@rdivide,median(Y_sums,1),Y_sums((np-pr-mnp+1):(np-pr))),[1 1 mnp length(sets)]));

                else
                    Y1 = Y_dark;
                end


                % % Normalize and log transform projections
                Y1 = real(-log(bsxfun(@rdivide,Y1,air_dark)));
                Y1(Y1 < 0) = 0;
                Y1(~isfinite(Y1)) = 0;

                % pad to nearest multiple of 8
                Y_1 = zeros(8*ceil(nu_c/8),8*ceil(nv_c/8),mnp,length(sets)); 
                Y_1(1:nu_c,1:nv_c,:,:) = Y1;

                % Append into output array
                Y_out(:, :, (np-pr-mnp+1):(np-pr)) = single(Y_1);

                clearvars Y1 Y_1

            end

        Y1 = Y_out;

            % Correct for blank projections
        if CTX.goremove  
            ind = Y_sums <  8*std(Y_sums);
            inds = find(ind == 1);
            if ~isempty(inds) && CTX.goremove
                fprintf('\tCorrecting for blank projections...\n')

                if any(diff(inds) == 1)
                    % Multiple blank projections in a row
                    mask = ones(size(Y1));
                    mask(:,:,ind) = 0;

                    % Set up filter based on the maximum number of blank projections in a row
    %                 sigma = 
                else
                    % For individual blank projections
                    for z = 1:length(inds)    
                        Y1(:,:,inds(z)) = 0; %( Y1(:,:,inds(z)-1) + Y1(:,:,inds(z)+1) )/2;
                    end
                end
            end

        end

            fprintf('\tSaving corrected projections...\n')
            save_nii(make_nii(single(Y1)),y_file);




end

fprintf('\tTime to get projections:\t%0.2f seconds\n', toc(nt))


end
