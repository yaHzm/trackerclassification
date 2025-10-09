function error_code = RandomlyPlaceTrackers(version_id, used_type_ids, use_last_data, plot_tracker, plot_trial)

% Author: Harald Hoppe
% Date: 2025-10-08

% Input variables:
%
% version_id    ... So far, this script only support version ID 4!
% used_type_ids ... a vector with the type IDs of the trackers that should be randomly placed in space, e.g. 1:4 or [10, 17, 26]
% use_last_data ... true if you want to use the same randomly placed trackers again
% plot_tracker  ... true if you want that the chosen tracker types (used_type_ids) are plotted
% plot_trial    ... true if you want to see a perspective image of the randomly placed trackers

% The following layout is supported:
%
%            Version 1            
%              / \               
%            0/   \2             
%            /     \            
%           /       \            
%          /         \           
%  side2  / 1       1 \  side1 
%        /             \           
%       /               \         
%      /                 \        
%    2/                   \0       
%    /          1          \    
%    -----------------------      
%         0   side0   2           
%
% There are 27 possible unique combinations:
% 000 / 001 / 002 / 010 / 011 / 012 / 020 / 021 / 022
% 100 / 101 / 102 / 110 / 111 / 112 / 120 / 121 / 122
% 200 / 201 / 202 / 210 / 211 / 212 / 220 / 221 / 222
%
% Be careful: In order to be able to get rid of rotational symmetry, side0 is always coded "inversely" which means
% that two LEDs are used on this side and the non-existent third defines the code for side0.
% Example: Code 111 means that LEDs 0 and 2 are present on side0, LEDs 1 are present on sides1 and side2.


% --------------------------------------------------------------------
% Initialize output variables
% --------------------------------------------------------------------
error_code = 0;


% --------------------------------------------------------------------
% Changeable constants
% --------------------------------------------------------------------
tri_side_length_in_mm = 64.0;  % triangle side length of the trackers (change this if you want to use smaller or larger trackers)
num_extra_led         = 0;     % the number of extra blobs in the image (reflections or other effects)
noise_in_mm           = 0.0;   % standard deviation of noise added to the LED positions
num_try               = 10000; % the number of random placements of the chosen trackers
max_rot_angle_in_deg  = 85.0;  % the maximum rotation angle of the trackers (with respect to the viewing direction)

x_min = -100.0; % minimal x-coordinate where the tracker should be placed
x_max =  100.0; % maximal x-coordinate where the tracker should be placed
y_min = -100.0; % minimal y-coordinate where the tracker should be placed
y_max =  100.0; % maximal y-coordinate where the tracker should be placed
z_min =  150.0; % minimal z-coordinate where the tracker should be placed
z_max =  200.0; % maximal z-coordinate where the tracker should be placed


% --------------------------------------------------------------------
% Check input variables
% --------------------------------------------------------------------
if nargin < 3
    use_last_data = false;
end

if nargin < 4
    plot_tracker = false;
end

if nargin < 5
    plot_trial = false;
end


% --------------------------------------------------------------------
% Constants
% --------------------------------------------------------------------
trafo_BE = [1.0, -1 / sqrt(3); ...
            0.0,  2 / sqrt(3)]; % transformation from equilateral coordinates to barycentric


% --------------------------------------------------------------
% Initial definitions and precalculations
% --------------------------------------------------------------
if version_id == 4
    rs_val_E = [ 0.25, 0.50,  0.75; ...
                -0.02, 0.02, -0.02]; % r and s values relative to the equilateral triangle

    num_led = 7;
else
    disp('Unknown version ID. Please check!');
    error_code = 1;
    return;
end


% --------------------------------------------------------------
% Calculate type enums and type IDs
% --------------------------------------------------------------
num_code_ids = size(rs_val_E, 2);
num_types    = num_code_ids * num_code_ids * num_code_ids; % the number of possible types for this number of codings LEDs
type_ids     = zeros(num_types, 3);
type_enum    = zeros(num_types, 1);
type_count   = 0;

for c0_id = 0 : num_code_ids - 1
    for c1_id = 0 : num_code_ids - 1
        for c2_id = 0 : num_code_ids - 1
            type_count = type_count + 1;
            type_ids(type_count, :) = [c0_id, c1_id, c2_id];
            type_enum(type_count)   = c0_id * 100 + c1_id * 10 + c2_id;

            disp([num2str(type_count), ': ', num2str([c0_id, c1_id, c2_id])]);
        end
    end
end


% --------------------------------------------------------------------
% Define tracker
% --------------------------------------------------------------------
num_tracker = length(used_type_ids);
num_pos     = num_led * num_tracker + num_extra_led;
led_pos_T   = cell(1, num_tracker);

p0 = [0.0; 0.0; 0.0];
p1 = tri_side_length_in_mm * [1.0;           0.0; 0.0];
p2 = tri_side_length_in_mm * [0.5; 0.5 * sqrt(3); 0.0];

for t_id = 1 : num_tracker
    % This is the same for all trackers
    led_pos_T{t_id} = zeros(3, num_led);
    led_pos_T{t_id}(:,1) = p0;
    led_pos_T{t_id}(:,2) = p1;
    led_pos_T{t_id}(:,3) = p2;
    next_l_id = 4;

    % This is the coding
    cur_type_id = used_type_ids(t_id);

    if cur_type_id > num_types
        disp('Unknown type ID. Please check!');
        error_code = 3;
        return;
    end

    for s_id = 0 : 2
        if s_id == 0
            q0 = p0; d1 = p1 - p0; d2 = p2 - p0;
        elseif s_id == 1
            q0 = p1; d1 = p2 - p1; d2 = p0 - p1;
        else
            q0 = p2; d1 = p0 - p2; d2 = p1 - p2;
        end

        cur_code_id = type_ids(cur_type_id, s_id +1);

        for c_id = 0 : 2
            % For side0: add all LEDs except the given one / for side1 and side2: add the given one
            if (s_id == 0 && c_id ~= cur_code_id) || (s_id ~= 0 && c_id == cur_code_id)
                rs_val_B = trafo_BE * rs_val_E(:, c_id +1);
                led_pos_T{t_id}(:, next_l_id) = q0 + rs_val_B(1) * d1 + rs_val_B(2) * d2;
                next_l_id = next_l_id + 1;
            end
        end        
    end

    if plot_tracker
        % Check plot
        figure(1); clf; hold on;
        cur_led_pos_T = led_pos_T{t_id};
        plot3(cur_led_pos_T(1, [1 2 3 1]), cur_led_pos_T(2, [1 2 3 1]), cur_led_pos_T(3, [1 2 3 1]), '-k');
        plot3(cur_led_pos_T(1,:), cur_led_pos_T(2,:), cur_led_pos_T(3,:), '.r', 'MarkerSize', 30);
        for l_id = 1 : num_led
            cur_pos = cur_led_pos_T(:, l_id);
            text(cur_pos(1), cur_pos(2) + 0.05 * tri_side_length_in_mm, cur_pos(3), [num2str(cur_pos(1), '%.4f'), ' / ', num2str(cur_pos(2), '%.4f')], 'HorizontalAlignment', 'center');
        end
        axis equal; grid off; axis off;
        xlim(tri_side_length_in_mm * [-0.15 1.15]); ylim(tri_side_length_in_mm * [-0.15, 0.5 * sqrt(3) + 0.15]);
        title(['Combination ', num2str(t_id), ':   ' num2str(type_ids(cur_type_id, :))]);
        drawnow; pause
    end
end



% --------------------------------------------------------------------
% Loop over all random placements
% --------------------------------------------------------------------
for try_id = 1 : num_try
    % --------------------------------------------------------------------
    % Randomly place trackers
    % --------------------------------------------------------------------
    led_pos_W = zeros(3, num_pos);
    
    for t_id = 1 : num_tracker
        tra_WT = [x_min + rand * (x_max - x_min); ...
                  y_min + rand * (y_max - y_min); ...
                  z_min + rand * (z_max - z_min)];
        
        dir_vec = tra_WT / norm(tra_WT);
        
        % Be careful: The trackers are defined such that their front surface is facing in positive
        % z-direction (in T-coordinates). Therefore, we have to align the negative z-direction of
        % the tracker with the direction vector dir_vec, which points approx. in positive z-direction
        % of the camera (W-coordinates).
        rot1 = GetRotationFromDirectionAndCosAngle(cross([0; 0; -1], dir_vec), -dir_vec(3));
        
        % rotate direction vector
        help_vec = [rand; rand; 0.0];
        rot_vec = cross(help_vec, dir_vec);
        rot_ang = rand * max_rot_angle_in_deg * pi / 180;
        rot_WT = GetRotationFromDirectionAndAngle(rot_vec, rot_ang);
        
        lower_id = (t_id-1) * num_led + 1;
        upper_id = t_id * num_led;
        led_pos_W(:, lower_id : upper_id) = rot_WT * rot1 * led_pos_T{t_id} + tra_WT * ones(1, num_led) + noise_in_mm * randn(3,num_led);
    end
    
    % Check if one triangle covers another LED
    found = false;
    for t0_id = 1 : num_tracker
        % These are the corners of the triangle that could cover.
        s_id = (t0_id-1) * num_led;
        q0 = led_pos_W(:,s_id+1);
        q1 = led_pos_W(:,s_id+2);
        q2 = led_pos_W(:,s_id+3);
        
        for t1_id = 1 : num_tracker
            if t1_id == t0_id, continue; end
    
            for l_id = 1 : num_led
                % This is a specific LED that could be covered.
                cur_led = led_pos_W(:, (t1_id-1) * num_led + l_id);
    
                % Intersect line from origin to LED with triangle plane and calculate barycentric coordinates
                res = [q1 - q0, q2 - q0, -cur_led] \ -q0;
                r = res(1); s = res(2); t = res(3);
    
                % Inside triangle and intersection in front of LED?
                if r >= 0.0 && s >= 0.0 && r + s <= 1.0 && t >= 0.0 && t <= 1.0
                    found = true;
                    %disp(['LED ', num2str((t1_id-1) * num_led + l_id), ' is covered by triangle ', num2str(t0_id)]);
                end
            end
        end
    end

    if found, continue; end
    
    
    % Add extra LEDs
    for p_id = 1 : num_extra_led
        led_pos_W(:, p_id + num_led * num_tracker) = [x_min + rand * (x_max - x_min);  y_min + rand * (y_max - y_min); z_min + rand * (z_max - z_min)];
    end
    
    % Save data
    if ~use_last_data
        fid = fopen('LEDPosData.dat', 'w');
        for p_id = 1 : num_pos
            fwrite(fid, led_pos_W(:,p_id), 'double');
        end
        fclose(fid);
    else
        fid = fopen('LEDPosData.dat', 'r');
        for p_id = 1 : num_pos
            led_pos_W(:,p_id) = fread(fid, 3, 'double');
        end
        fclose(fid);
    end
    
    if plot_trial
        % Plot constellation
        figure(2); clf; hold on;
        plot3(led_pos_W(1,:), led_pos_W(2,:), led_pos_W(3,:), '.w', 'MarkerSize', 10);
        for l_id = 1 : size(led_pos_W, 2)
            %text(led_pos_W(1,l_id) + 3.0, led_pos_W(2,l_id), led_pos_W(3,l_id), num2str(l_id), 'Color', 'w');
        end
        axis equal; grid on;
        set(gca, 'Color', 'k', 'GridColor', 'w');
        camproj('perspective'); campos([0.0, 0.0, 0.0]); camup([0.0, -1.0, 0.0]);
        drawnow;
        pause
    end
end % end of loop over all trials












