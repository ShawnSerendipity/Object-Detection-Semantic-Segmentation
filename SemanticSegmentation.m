%% ============================ Initialization ============================
clear; close all; clc

%% ==================== Load Pretrained Neural Network ====================
vgg16();

%% ========================== Load Training Sets ==========================
% Load CamVid Images
imds = imageDatastore('/Users/shawn/Desktop/Semantic Segmentation/701_StillsRaw_full');
% I = readimage(imds,3);
% I = histeq(I); imshow(I)

% Load CamVid Pixel-Labeled Images
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

labelIDs = camvidPixelLabelIDs();

pxds = pixelLabelDatastore('/Users/shawn/Desktop/Semantic Segmentation/LabeledApproved_full',classes,labelIDs);

% C = readimage(pxds,3);
% cmap = camvidColorMap;
% B = labeloverlay(I,C,'ColorMap',cmap);
% imshow(B)
% pixelLabelColorbar(cmap,classes);

%% ============================ Analyze Dataset ===========================
% Visualize the distribution of class labels in the CamVid dataset
tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% ============= Prepare Training, Validation, and Test Sets ==============
% Randomly splits the image and pixel label data into a training, validation and test set.
% Where Training Sets 60%, Validation 20% and Test Sets 20%
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);
% numTrainingImages = numel(imdsTrain.Files)
% numValImages = numel(imdsVal.Files)
% numTestingImages = numel(imdsTest.Files)

%% ========================== Create the Network ==========================
% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [720 960 3];

% Specify the number of classes.
numClasses = numel(classes);

% Create SegNet Network
lgraph = segnetLayers(imageSize,numClasses,'vgg16');

% lgraph initially has 91 Layers. 
lgraph.Layers

% Plot the 91-Layer lgraph 
fig1=figure('Position', [100, 100, 1000, 1100]);
subplot(1,2,1)
plot(lgraph);
axis off
axis tight
title('Complete Layer Graph')
subplot(1,2,2)
plot(lgraph);
xlim([2.862 3.200])
ylim([-0.9 10.9])
axis off 
title('Last 9 Layers Graph')

%% ================ Balance Classes Using Class Weighting =================
% Get the imageFreq using the data from the countEachLabel function
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;

% The higher the frequency of a class the smaller the classWeight
classWeights = median(imageFreq) ./ imageFreq

%% =================== Modify VGG16 Model For Our Task ====================
% Create a new layer with the new pixelClassificationLayer. 
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

% Remove last layer of vgg16 and add the new one we created. 
lgraph = removeLayers(lgraph, {'pixelLabels'});
lgraph = addLayers(lgraph, pxLayer);

% Connect the newly created layer with the graph. 
lgraph = connectLayers(lgraph, 'softmax','labels');
lgraph.Layers
subplot(1,2,2)
plot(lgraph);
xlim([2.862 3.200])
ylim([-0.9 10.9])
axis off 
title(' Modified last 9 Layers Graph')

%% ======================= Select Training Options ========================
% The optimization algorithm used for training is Stochastic Gradient Decent with Momentum (SGDM).
options = trainingOptions('sgdm', ... % This is the solver's name; sgdm: stochastic gradient descent with momentum
    'Momentum', 0.9, ...              % Contribution of the gradient step from the previous iteration to the current iteration of the training; 0 means no contribution from the previous step, whereas a value of 1 means maximal contribution from the previous step.
    'InitialLearnRate', 1e-2, ...     % low rate will give long training times and quick rate will give suboptimal results 
    'L2Regularization', 0.0005, ...   % Weight decay - This term helps in avoiding overfitting
    'MaxEpochs', 120,...              % An iteration is one step taken in the gradient descent algorithm towards minimizing the loss function using a mini batch. An epoch is the full pass of the training algorithm over the entire training set.
    'MiniBatchSize', 4, ...           % A mini-batch is a subset of the training set that is used to evaluate the gradient of the loss function and update the weights.
    'Shuffle', 'every-epoch', ...     % Shuffle the training data before each training epoch and shuffle the validation data before each network validation.
    'Verbose', false,...        
    'Plots','training-progress');  

%% ========================== Data Augmentation ===========================
% Use the imageDataAugmenter to specify these data augmentation parameters
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);

%% =========================== Start Training =============================
% Combine the training data and data augmentation selections
pximds = pixelLabelImageSource(imdsTrain,pxdsTrain,'DataAugmentation',augmenter);

% Trains the network for semantic segmentation   
[net, info] = trainNetwork(pximds,lgraph,options);
disp('NN trained');

%% ===================== Test and Evaluate Network ========================
% Read the image and do semantic segmantation
I = readimage(imdsTest,5);
C = semanticseg(I, net);

% Show the result
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);

% Compare the result with original labeled image
% The green and magenta regions highlight areas where the segmentation results differ from the expected ground truth.
expectedResult = readimage(pxdsTest,35);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

% Use the intersection-over-union (IoU) metric(Jaccard index) to measure the amount of overlap per class.
iou = jaccard(C,expectedResult);
table(classes,iou)

% Measure accuracy for multiple test images
pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
metrics.DataSetMetrics
metrics.ClassMetrics

%% ==================== Apply Our Model To New Image ======================
% Read the image and do semantic segmantation
I = readimage('');
C = semanticseg(I, net);

% Show the result
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);
