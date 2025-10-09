function rot = GetRotationFromDirectionAndCosAngle(d, cosAngle, normalizeDirection)

% Author: Harald Hoppe

% Input parameters
% d        ... rotation axis
% cosAngle ... cos of rotation angle (since sinAngle is always considered
% positive, angle lies between 0 and pi. Inverting dir results in the
% missing angle region)

% Output parameters
% rot ... rotation matrix

if nargin < 3
    normalizeDirection = true;
end

if normalizeDirection
    help = d(1) * d(1) + d(2) * d(2) + d(3) * d(3);
    
    if help == 0.0
        rot = eye(3);
        return;
    else
        d = d / sqrt(help);
    end
end

sinAngle = sqrt(1 - cosAngle * cosAngle);
t = 1.0 - cosAngle;

rot(1, 1) = t * d(1) * d(1) + cosAngle;
rot(1, 2) = t * d(1) * d(2) - sinAngle * d(3);
rot(1, 3) = t * d(1) * d(3) + sinAngle * d(2);
rot(2, 1) = t * d(2) * d(1) + sinAngle * d(3);
rot(2, 2) = t * d(2) * d(2) + cosAngle;
rot(2, 3) = t * d(2) * d(3) - sinAngle * d(1);
rot(3, 1) = t * d(3) * d(1) - sinAngle * d(2);
rot(3, 2) = t * d(3) * d(2) + sinAngle * d(1);
rot(3, 3) = t * d(3) * d(3) + cosAngle;

	