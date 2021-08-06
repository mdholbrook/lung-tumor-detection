function Y = genprojs_synth_lung_tumors_v2( X, RP, outpath, recon_param_file )
% recon_synth_lung_tumors: 
%   v2: reads in parameters for projecting into the same geometry as
%   original projections

% RP = loadjson(recon_param_file);
saved_dir = fileparts(recon_param_file);

X = single(X);

    %% Load recon geometry

    
    cal_file = fullfile(saved_dir, 'geo_calibration.cal');
    p = load(cal_file, '-mat');
    params = p.params(1:9);
    %params = [820.8677      679.2406       517.516      402.5594    0.01075158    0.03674691    0.01109152     -8.288696             0];
    
    
    %% Manage projection cropping
    
    % Update projection size based on input crop parameters
    xposition = RP.xposition;
    zcrop = [xposition(2), sum(xposition([2 4]))-1];

    % Adjust Z-offset to center reconstruction
    M1 = params(1)/params(2);
    zo1 = -RP.dv1 * (params(4) - mean(zcrop)) / M1;

    params(8) = zo1;

    % Update geometry
    xcrop = [xposition(1), sum(xposition([1 3]))-1];
    zcrop = [xposition(2), sum(xposition([2 4]))-1];

    xcrop_d = [ xposition(1)-1 xcrop(2) ];
    zcrop_d = [ xposition(2)-1 zcrop(2) ];
    
    
    % Update calibration parameters using the cropped values
    params(3) = params(3) - xcrop_d(1);
    params(4) = params(4) - zcrop_d(1);


    %% Set up recon parameters
    geo_all1 = [outpath, 'geo.geo'];
    du = 0.075;
    dv = 0.075;
    rotdir = 1;
    np = RP.np;
    ap = RP.rotation1/np;
    angles1 = single(0:ap:(RP.rotation1 - ap));
    weight1 = ones(1,np);
    ao = 0;
    nsets = 1;

    geo_all_projs_v4( geo_all1, params, du, dv, rotdir, np, angles1, ao, nsets, np, weight1);

    % Projection sizes
    nu_c = length(xcrop(1):xcrop(2)); %length(305:1384);
    nv_c = length(zcrop(1):zcrop(2)); %length(410:1002);

    nx = RP.nx1;
    ny = RP.ny1;
    nz = RP.nz1;
    sz = [nx ny nz];

    dx = 0.063;
    dy = 0.063;
    dz = 0.063;

    zo = params(8);

    [x, y] = meshgrid((0:(sz(1)-1))-sz(1)./2,(0:(sz(2)-1))-sz(2)./2);
    dist = sqrt(single(x).^2 + single(y).^2);
    mask = dist < (min(sz(1)./2,sz(2)./2) - 2); % make sure we stay inside the GPU mask
    mask = repmat(mask,[1 1 sz(3)]);
    mask = single(mask);
    mask = single(mask(:));
    clear x y dist;

    scale = 1; % scalar volume multiplier
    np = np;
    
    % Flags
    FBP = 1; % (1) FBP (Hann filter), (0) simple backprojection
    simple_back_projection = 0;
    implicit_mask = 1; % (0) - do not use implicit reconstruction mask; (1) use implicit reconstruction mask
    explicit_mask = 0; % (0) - do not use explicit reconstruction mask; (1) use explicit reconstruction mask
    use_affine = 1;    % (0) - affine transform is not used (even if one is provided); (1) - affine transform is used


    % new DD operators
    nu = 8*ceil(nu_c/8);
    nv = 8*ceil(nv_c/8);
%     int_params = int32([nu nv np nx ny nz 0]); % unfiltered
%     int_paramsf = int32([nu nv np nx ny nz 1]); % filtered
    int_params =   int32([nu nv np nx ny nz simple_back_projection use_affine]);
    int_paramsf = int32([nu nv np nx ny nz FBP                    use_affine]);
    double_params = [du dv dx dy dz zo ao params(1:7)];


    double_params1 = [du dv dx dy dz zo ao params(1:7) scale];

    % Make a filter for FDK
    filter_type = 'ram-lak';

    scaling = 1;
    lenu = 2^(ceil(log(nu+1.0)/log(2.0))); % next power of 2...according to Kak and Slaney, should probably double again
    filt_name = [outpath filter_type '_' num2str(lenu) '.txt'];
    cutoff = 1;
    detector_type = 0; % flat panel
    rmask = uint16(0);
    make_filter_fessler( filt_name, du, lenu, params(1), 'ram-lak', detector_type, cutoff );

    % Affine transform calculated 12/11/2017
    Rt_aff1 = single([1 0 0 0 1 0 0 0 1 0 0 0]);
    
    gpu_list = int32(0);
    
    clear DD_init;
    recon_alloc = DD_init(int_paramsf(:), double_params1(:), filt_name, geo_all1, Rt_aff1(:), rmask, gpu_list);

    
    R1 =  @(x,w)  DD_project(x, recon_alloc, w);

    Rtf1 = @(y,w) DD_backproject(y, recon_alloc, int32(0), w);

    %% Forward project

    Y = R1(X(:), weight1);

    Y = reshape(Y, [nu, nv, np]);

%     savefile = sprintf('%s/Y%d.nii', outpath, j);
%     save_nii(make_nii(single(Y)), savefile); 
    
%     Y_all(:, :, :, j) = Y;

    %% Back project

%     X1 = Rtf1(Y(:), Rt_aff1);
% 
%     X1 = reshape(X1, [nx, ny, nz]);
%     
    

end

