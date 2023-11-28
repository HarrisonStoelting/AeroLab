clear
clc

% Specify the file path of the TIFF image
filePath = '0deg_supersonic_40psi.tif';

% Open the TIFF file
tiffObj = Tiff(filePath, 'r');

% Read the image data
imageData = read(tiffObj);
figure(1); imagesc(imageData); colormap('parula');
figure(2); plot(imageData(320,:), 'Linewidth', 3); hold all;

fprintf(imageData(320,:));

% Display the image
imshow(imageData);

% Close the TIFF file
close(tiffObj);

