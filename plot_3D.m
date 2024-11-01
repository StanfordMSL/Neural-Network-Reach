clear all
clc

% load in learned PWA function
load('models/quadratic/quadratic_pwa.mat');
num_regions = length(A);

% for frame = 1:num_regions
%     for i = 1:frame
%         P(i) = Polyhedron(A{i}, b{i});
%         P(i).addFunction(AffFunction(C{i}(1,:), d{i}(1)), 'g')
%     end
% 
%     U = PolyUnion('Set', P, 'Overlaps', false, 'Convex', true)
% 
%     f = figure;
%     f.Color = [1 1 1]; 
%     U.fplot('show_set', false, 'colororder', 'fixed', 'LineWidth', 0.25)
%     grid off
%     axis off
%     xlim([-1, 1])
%     ylim([-1 1])
%     zlim([-1 2])
%     filename = "figures/enum_gif/" + string(frame) + ".png";
%     saveas(gcf, filename)
%     close all
% end



% Set up file names and output GIF
pngFiles = dir('figures/enum_gif/*.png'); % Adjust the directory and file pattern if necessary
outputFile = 'figures/enum_gif/animated.gif';

% Loop through PNG files to create GIF
for i = 1:202
    % Read the current image
    filename = strcat('figures/enum_gif/', string(i), '.png')
    img = imread(filename);

    % Convert to indexed image with colormap
    [indexedImg, cmap] = rgb2ind(img, 256);

    % Write to the GIF file
    if i == 1
        % For the first image, create the GIF file
        imwrite(indexedImg, cmap, outputFile, 'gif', 'LoopCount', Inf, 'DelayTime', 0.1);
    else
        % For subsequent images, append to the GIF file
        imwrite(indexedImg, cmap, outputFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
end
