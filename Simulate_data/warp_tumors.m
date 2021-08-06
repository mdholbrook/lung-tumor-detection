function tumor = warp_tumors( tumor, shear_params )
%warp_tumors: warps 3D tumor crops to generate training data
% Args:
%   tumor: 3D tumor image or tumor mask
%   shear_params: min and max of random shearing

if length(unique(tumor)) < 3
    mask = true;
else
    mask = false;
end

% Get constants for random shear ranges
a = shear_params(1);
b = shear_params(2) - a;

% Calculate random skews
Sx = a + b*rand;
Sy = a + b*rand;
Sz = a + b*rand;
Stform = [Sx 0 0 0; 0 Sy 0 0; 0 0 Sz 0; 0 0 0 1];

% Calcuate random rotations
tx = 2*pi*rand;
ty = 2*pi*rand;
tz = 2*pi*rand;
XRtform = [1 0 0 0; 0 cos(tx) sin(tx) 0; 0 -sin(tx) cos(tx) 0; 0 0 0 1];
YRtform = [cos(ty) 0 -sin(ty) 0; 0 1 0 0; sin(ty) 0 cos(ty) 0; 0 0 0 1];
ZRtform = [cos(tz) sin(tz) 0 0; -sin(tz) cos(tz) 0 0; 0 0 1 0; 0 0 0 1];

% Combine transforms
tform_mat = Stform * XRtform * YRtform * ZRtform;
tform = affine3d(tform_mat);

% Transform the tumor
tumor = imwarp(double(tumor), tform);

% Re-mask
if mask
    tumor = tumor > 0.5;
end

end

