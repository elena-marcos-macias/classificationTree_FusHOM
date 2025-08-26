%--------- Add function path and set save path ----------
addpath(genpath('./data'));
addpath (genpath('./utils'));
addpath ('./requirements');
savePath = './results/';

% --------- Read instructions JSON archive -----------
json = readstruct("data/instructionsDecisionTree.json");


%% TRAINING DATASET
fileName = char(json.inputDataSelection.fileName);
T_Original = rmmissing(readtable(['./data/' fileName])); % Load and Remove rows that contain at least one NaN


    % === 1. All PREDICTOR variables ===
    T_Data = selectColumns (T_Original,...
        json.inputDataSelection.columnCriteria.target_columns, ...
        json.inputDataSelection.columnCriteria.ignore_columns);

    catVariable = char(json.inputDataSelection.catVariable); % Identify categorical variables as such

    % === 2. RESULTS variable ===
    T_ResultsVariable = T_Original.(json.inputDataSelection.resultsVariable);

    % Identify minority and majority classes
    [classNames, ~, idxClass] = unique(T_ResultsVariable);
    counts = accumarray(idxClass,(ones(numel(T_ResultsVariable),1)));
  
    if max(counts) ~= min(counts)
        [~, majorityIdx] = max(counts);
        [~, minorityIdx] = min(counts);
    elseif max(counts) == min(counts)
        [~, majorityIdx] = max(counts);
        minorityIdx = majorityIdx+1;
    end
    
    majorityClass = string(classNames(majorityIdx));
    minorityClass = string(classNames(minorityIdx));



%% TRAIN THE DECISION TREE - TRAINING DATASET
% This section creates a Classification Tree model, trained with the whole
% dataset informing us of the importance of the predictor variables. It
% displays a decision tree graph and a predictors' importance bar graph and 
% table


%    % Try (B) high-res PNG via print (classic figures)
%    try
%        print(treeFig, fullfile(savePath, 'decisionTree.png'), '-dpng', '-r300');
%    catch
%        % Try (C) UIFigure export (App Designer-style windows)
%        if exist('exportapp', 'file') == 2
%            exportapp(treeFig, fullfile(savePath, 'decisionTree.png'));
%        else
%            % Try (D) last-resort snapshot
%            fr = getframe(treeFig);
%            imwrite(fr.cdata, fullfile(savePath, 'decisionTree.png'));
%        end
%    end
%end

% ------------------ 1) TREE ------------------ 
Mdl = fitctree(T_Data, T_ResultsVariable, ...
    'CategoricalPredictors', {catVariable}, 'MinParentSize', 3);

    % Open the (GUI) tree viewer
    view(Mdl, 'Mode', 'graph');
    drawnow; pause(0.1);  % give MATLAB a beat to create the window
    
    % Get the most recently created figure (the tree viewer)
    allFigs = findall(groot, 'Type', 'figure');
    treeFig = allFigs(1);   % newest figure
    
    % Save vector PDF via exportgraphics
        exportgraphics(treeFig, fullfile(savePath, char(json.outputFileNames.decisionTree)), 'ContentType', 'vector');


% ------------------ 2) PREDICTORS' IMPORTANCE ------------------ 
imp = predictorImportance(Mdl);
predictorNames = Mdl.PredictorNames;

%To visualize the importance as a graph
figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Importance');
xlabel('Predictors');

ax = gca;
ax.XTick = 1:numel(predictorNames);
ax.XTickLabel = predictorNames;
ax.XTickLabelRotation = 45;
ax.TickLabelInterpreter = 'none';

savefig(fullfile(savePath, char(json.outputFileNames.predictorImportance)));

%To visualize the importance table
PredictorImportanceTable = table(predictorNames', imp', 'VariableNames', {'Predictor', 'Importance'});
disp('PredictorImportance_Table:');
disp(PredictorImportanceTable);



%% CROSS-VALIDATION METHODS
% To validate the model, and since we don't have a different dataset from
% the training dataset to try it with, we are going to apply
% cross-validation methods. Since the metrics' variability is very high the
% experiment will be performed nRuns times. 1. Randomiced K-Fold method. 
% 2. Weighted K-fold to manage imbalance between minority and majority
% classes. 3. Oversampled K-fold to manage imbalance.

% Numer of folds for K-fold
nFolds = json.crossValidationMethods.nFolds;

% Number of repetitions
nRuns = json.crossValidationMethods.nRuns;

% Create string with the results' labels and a logical vector for those results'
% labels that coincide with the % minority class, in this case 'Dead'
trueLabels = string(T_ResultsVariable);
trueBinary = trueLabels == minorityClass{1};

% Preallocate result matrices: 
errorResults = zeros(nRuns, 4, 3);  % [run, metric, model] -> metric = [Overall, Maj, Min, AUC]
confusionmatResults = zeros(2,2,nRuns,3); % [2x2 confusion matrix (actual class, predicted class), run, model]
modelNames = {'Standard', 'Weighted', 'Oversampled'};

% New: store FP, TP (variable-length) in cell arrays and AUCs in numeric matrix
FPAll = cell(nRuns, 3);   % FPAll{run, model} = fp vector
TPAll = cell(nRuns, 3);   % TPAll{run, model} = tp vector
AUCs  = zeros(nRuns, 3);  % AUCs(run, model) = auc

for run = 1:nRuns
    fprintf('Running iteration %d of %d...\n', run, nRuns);

    % ----- 1. Standard CV Tree -----
    CVMdl = fitctree(T_Data, T_ResultsVariable, ...
        'KFold',           nFolds, ...
        'CategoricalPredictors', {catVariable}, ...
        'MinParentSize',   3);

    [Label, Score] = kfoldPredict(CVMdl);
    missClassRate = kfoldLoss(CVMdl);
    [missMajority, missMinority] = classwiseMisclassification(T_ResultsVariable, Label, majorityClass, minorityClass);
    [fp1,tp1,~,auc1] = perfcurve(trueBinary, Score(:,2), 1);

    % save metrics
    errorResults(run,:,1) = [missClassRate, missMajority, missMinority, auc1];
    cm_Label = string(Label);
    cmSt = confusionmat(trueLabels, cm_Label);
    confusionmatResults(:,:,run,1) = cmSt;

    % store FP/TP/AUC
    FPAll{run,1} = fp1(:);   % ensure column
    TPAll{run,1} = tp1(:);
    AUCs(run,1)  = auc1;

    % ----- 2. Weighted CV Tree -----
    WeightCVMdl = fitctreeWeightCV(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    [wtLabel, wtScore] = kfoldPredict(WeightCVMdl);
    missClassRateWeight = kfoldLoss(WeightCVMdl);
    [missMajorityW, missMinorityW] = classwiseMisclassification(T_ResultsVariable, wtLabel, majorityClass, minorityClass);
    [fp2,tp2,~,auc2] = perfcurve(trueBinary, wtScore(:,2), 1);

    errorResults(run,:,2) = [missClassRateWeight, missMajorityW, missMinorityW, auc2];
    cm_wtLabel = string(wtLabel);
    cmWt = confusionmat(trueLabels, cm_wtLabel);
    confusionmatResults(:,:,run,2) = cmWt;

    % store FP/TP/AUC
    FPAll{run,2} = fp2(:);
    TPAll{run,2} = tp2(:);
    AUCs(run,2)  = auc2;

    % ----- 3. Oversampled CV Tree -----
    [OSLabels, OSScores] = kfoldPredictOS(T_Data, T_ResultsVariable, nFolds, {catVariable}, 3);
    missClassRateOS = sum(~strcmp(OSLabels, T_ResultsVariable)) / numel(T_ResultsVariable);
    [missMajorityOS, missMinorityOS] = classwiseMisclassification(T_ResultsVariable, OSLabels, majorityClass, minorityClass);
    [fp3,tp3,~,auc3] = perfcurve(trueBinary, OSScores(:,2), 1);

    errorResults(run,:,3) = [missClassRateOS, missMajorityOS, missMinorityOS, auc3];
    cm_OSLabels = string(OSLabels);
    cmOS = confusionmat(trueLabels, cm_OSLabels);
    confusionmatResults(:,:,run,3) = cmOS;

    % store FP/TP/AUC
    FPAll{run,3} = fp3(:);
    TPAll{run,3} = tp3(:);
    AUCs(run,3)  = auc3;

end

% AFTER THE RUNS: find max, min and closest-to-median for each model
% Prepare containers to hold results
Results = struct();
for m = 1:3
    auc_vec = AUCs(:,m);

    % handle possible NaNs (if any)
    validIdx = ~isnan(auc_vec);
    if ~any(validIdx)
        warning('No valid AUCs for model %d (%s).', m, modelNames{m});
        Results(m).max = [];
        Results(m).min = [];
        Results(m).median_closest = [];
        continue
    end

    % index of maximum AUC
    [maxAuc, idxMax] = max(auc_vec(validIdx));
    % map back to absolute index
    validRuns = find(validIdx);
    idxMax = validRuns(idxMax);

    % index of minimum AUC
    [minAuc, idxMin] = min(auc_vec(validIdx));
    idxMin = validRuns(idxMin);

    % index closest to median AUC
    medAuc = median(auc_vec(validIdx));
    [~, relIdx] = min(abs(auc_vec(validIdx) - medAuc));
    idxMed = validRuns(relIdx);

    % Extract FP/TP and ensure they are double column vectors
    fp_max = double(FPAll{idxMax,m}(:));
    tp_max = double(TPAll{idxMax,m}(:));
    auc_max = double(AUCs(idxMax,m));

    fp_min = double(FPAll{idxMin,m}(:));
    tp_min = double(TPAll{idxMin,m}(:));
    auc_min = double(AUCs(idxMin,m));

    fp_med = double(FPAll{idxMed,m}(:));
    tp_med = double(TPAll{idxMed,m}(:));
    auc_med = double(AUCs(idxMed,m));
    
    % Store into Results struct
    Results(m).modelName = modelNames{m};

    Results(m).max.runIndex = idxMax;
    Results(m).max.auc = auc_max;
    Results(m).max.fp = fp_max;
    Results(m).max.tp = tp_max;

    Results(m).min.runIndex = idxMin;
    Results(m).min.auc = auc_min;
    Results(m).min.fp = fp_min;
    Results(m).min.tp = tp_min;

    Results(m).median_closest.runIndex = idxMed;
    Results(m).median_closest.auc = auc_med;
    Results(m).median_closest.fp = fp_med;
    Results(m).median_closest.tp = tp_med;

    % (Optional) show a quick summary
    fprintf('Model: %s  |  max AUC = %.4f (run %d), min AUC = %.4f (run %d), median-close AUC = %.4f (run %d)\n', ...
        modelNames{m}, auc_max, idxMax, auc_min, idxMin, auc_med, idxMed);
end



%% PLOT HISTOGRAMS PER CROSS-VALIDATION METHOD AND MEASURE
metrics = {'OverallError', 'MajorityError', 'MinorityError', 'AUC'};
modelNames = {'Standard', 'Weighted', 'Oversampled'};
colors = lines(3);
numBins = 50;

% --- 1. Find global axis limits across ALL histograms ---
globalMinX = inf;
globalMaxX = -inf;
globalMaxY = 0;

for m = 1:3
    for k = 1:4
        data = errorResults(:, k, m);

        % Temporary histogram to inspect bin edges and values
        htemp = histogram(data, numBins);
        globalMinX = min(globalMinX, min(htemp.BinEdges));
        globalMaxX = max(globalMaxX, max(htemp.BinEdges));
        globalMaxY = max(globalMaxY, max(htemp.Values));
        delete(htemp);
    end
end

% --- 2. Plot with the same global axis limits ---
figure('Name', 'All Metrics Distributions with Global Limits', 'NumberTitle', 'off');

for m = 1:3  % rows = models
    for k = 1:4  % columns = metrics
        ax = subplot(3, 4, (m-1)*4 + k);
        data = errorResults(:, k, m);

        histogram(data, numBins, ...
            'FaceColor', colors(m, :), ...
            'EdgeColor', 'k');

        % Add mean vertical line
        meanVal = mean(data);
        hold on;
        xline(meanVal, '--r', 'LineWidth', 2);
        hold off;

        % Apply same global axis limits
        xlim(ax, [globalMinX, globalMaxX]);
        ylim(ax, [0, globalMaxY]);

        if m == 1
            title(metrics{k}, 'FontSize', 12, 'FontWeight', 'bold');
        end
        if k == 1
            ylabel(modelNames{m}, 'FontSize', 12, 'FontWeight', 'bold');
        end

        grid on;
    end
end

sgtitle('Distribution of Errors per Metric and Model', 'Fontsize', 16);
savefig(fullfile(savePath, char(json.outputFileNames.crossValidationHistograms)));



%% TEST1 - TRAINING DATASET
% Obtain the models' predictions for this dataset
[test1Labels, test1Scores] = predict(Mdl, T_Data);

% Obtain the missclassification rate for all subjects in the dataset
missClassRateTest1 = sum(~strcmp(test1Labels, T_ResultsVariable)) / numel(T_ResultsVariable);

% Obtain the missclassification rates for each class separately (minority
% and majority)
[missMajorityTest1, missMinorityTest1] = classwiseMisclassification(T_ResultsVariable, test1Labels, majorityClass, minorityClass);

% Create a logical vector for those results' labels that coincide with the
% minority class, in this case 'Dead'
trueBinary = string(T_ResultsVariable) == minorityClass{1};

% Obtain the AUC
[fp4,tp4,~,aucTest1] = perfcurve(trueBinary, test1Scores(:,2), 1);



%% QUALITY METRICS TABLE
% For the 3 cross-validation models the table displays the mean value of 
% all iterations. For Test1 it displays the value of the test.

% Prepare summary table for errors
Models = {'Standard CV'; 'Weighted CV'; 'Oversampled CV'; 'Test1 - Training Data'};
Metrics = {'Mean_OverallError'; 'Std_OverallError'; 'Mean_MajorityError'; 'Std_MajorityError'; 'Mean_MinorityError'; 'Std_MinorityError'; 'Mean_AUC'; 'Std_AUC'};

means = squeeze(mean(errorResults,1))';
meanOS = means(:,1);
meanME = means(:,2);
meanmE = means(:,3);
meanAUC = means(:,4);

STDs = squeeze(std(errorResults,1))';
stdOS = STDs(:,1);
stdME = STDs(:,2);
stdmE = STDs(:,3);
stdAUC = STDs(:,4);
stdTest = ("-");

% Prepare columns
clmMeanOS = [meanOS;missClassRateTest1];
clmMeanME = [meanME;missMajorityTest1];
clmMeanmE = [meanmE;missMinorityTest1];
clmMeanAUC = [meanAUC;aucTest1];
clmStdOS = [stdOS;stdTest];
clmStdME = [stdME;stdTest];
clmStdmE = [stdmE;stdTest];
clmStdAUC = [stdAUC;stdTest];

Matrix = [clmMeanOS, clmStdOS, clmMeanME, clmStdME, clmMeanmE, clmStdmE, clmMeanAUC, clmStdAUC];
PlotMatrix = [clmMeanOS'; clmStdOS'; clmMeanME'; clmStdME'; clmMeanmE'; clmStdmE'; clmMeanAUC'; clmStdAUC'];

% Create table
ErrorTable = array2table(Matrix,...
    'VariableNames', Metrics,...
    'RowNames', Models);
disp('Error_Table:');
disp(ErrorTable);



%% CONFUSION CHART OF HISTOGRAMS - 2×8 LAYOUT
numBins = 10;

% Create one big tiledlayout (2 rows × 8 columns)
bigFig = figure('Name', 'All Confusion Matrix Histograms', 'NumberTitle', 'off',...
    'Position', [65,248.2,1415.2,353.6]);
tBig = tiledlayout(bigFig, 2, 8, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

% Define tile positions for each model's histograms (keeps your original layout)
tilePositions = { [1, 2, 9, 10], ... % Model 1
                  [3, 4, 11, 12], ... % Model 2
                  [5, 6, 13, 14] };   % Model 3
% Blank (for confusion counts/heatmap): [7, 8, 15, 16]

% --- Paso 0: Calcular límites globales ---
maxXGlobal = -inf;
maxYGlobal = -inf;
for m = 1:3
    for pos = 1:4
        switch pos
            case 1, row = 1; col = 1;
            case 2, row = 1; col = 2;
            case 3, row = 2; col = 1;
            case 4, row = 2; col = 2;
        end
        values = squeeze(confusionmatResults(row, col, :, m));
        [binCounts, binEdges] = histcounts(values, numBins);
        if ~isempty(binEdges)
            maxXGlobal = max(maxXGlobal, max(binEdges));
        end
        if ~isempty(binCounts)
            maxYGlobal = max(maxYGlobal, max(binCounts));
        end
    end
end

% Safety fallback if histcounts returned empty for any reason
if ~isfinite(maxXGlobal) || maxXGlobal == -inf
    maxXGlobal = 1;
end
if ~isfinite(maxYGlobal) || maxYGlobal == -inf
    maxYGlobal = 1;
end

% --- Paso 1: Dibujar histogramas usando límites globales ---
% Keep an axes handle for each of the 16 tiles so we can compute bounding boxes later
axesHandles = gobjects(16,1);

for m = 1:3
    for pos = 1:4
        tileIdx = tilePositions{m}(pos);
        ax = nexttile(tBig, tileIdx);
        axesHandles(tileIdx) = ax;
        
        switch pos
            case 1, row = 1; col = 1;
            case 2, row = 1; col = 2;
            case 3, row = 2; col = 1;
            case 4, row = 2; col = 2;
        end
        
        values = squeeze(confusionmatResults(row, col, :, m));
        
        % Color: blue for true (diagonal), red for false (off-diagonal)
        if pos == 1 || pos == 4
            faceColor = [0.2 0.6 0.8];
        else
            faceColor = [0.85 0.33 0.1];
        end
        
        histogram(ax, values, numBins, 'FaceColor', faceColor, 'EdgeColor', 'k');
        axis(ax, 'square');
        grid(ax, 'on');
        
        xlim(ax, [0, maxXGlobal]);
        ylim(ax, [0, maxYGlobal]);
        
        % Labels (same formatting as original)
        switch pos
            case 1
                ylabel(ax, majorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            case 3
                ylabel(ax, minorityClass, 'FontSize', 14, 'FontWeight', 'bold');
                xlabel(ax, majorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            case 4
                xlabel(ax, minorityClass, 'FontSize', 14, 'FontWeight', 'bold');
            otherwise
                % keep default ticks/labels for interior tiles
        end
    end
end

% Create placeholders for the remaining tiles (7,8,15,16) so they have axes handles
for idx = 1:16
    if isempty(axesHandles(idx)) || ~isgraphics(axesHandles(idx))
        ax = nexttile(tBig, idx);
        % keep placeholder visually neutral for now (we'll populate them below)
        cla(ax);
        axesHandles(idx) = ax;
    end
end

% --- Compute confusion matrix for test1 and display its values in tiles 7,8,15,16 ---
% Ensure test1Labels and true labels exist
if ~exist('test1Labels','var') || ~exist('T_ResultsVariable','var')
    error('Variables test1Labels or T_ResultsVariable not found in workspace.');
end

% Convert labels to cellstr for consistent display/labeling
try
    trueLabelsForCM = cellstr(T_ResultsVariable);
catch
    trueLabelsForCM = cellstr(string(T_ResultsVariable));
end
try
    predLabelsForCM = cellstr(test1Labels);
catch
    predLabelsForCM = cellstr(string(test1Labels));
end

% Compute confusion matrix (rows = true, cols = predicted)
confMatTest1 = confusionmat(trueLabelsForCM, predLabelsForCM);

% Get class names in the order used by confusionmat (stable order from true labels)
if exist('classNames','var') && numel(classNames) == size(confMatTest1,1)
    cmClassNames = classNames;
else
    cmClassNames = unique(trueLabelsForCM, 'stable');
end

[nRows, nCols] = size(confMatTest1);

% --- ASSUME 2x2 confusion matrix (per your request). Error if not 2x2. ---
if ~(nRows == 2 && nCols == 2)
    error('confMatTest1 is not 2x2. This script assumes a 2x2 confusion matrix.');
end

% Define colors
blueColor = [0.2 0.6 0.8];
redColor  = [0.85 0.33 0.1];

% Label offset values (normalized units relative to each axis)
% these ensure labels sit further away from tick labels, matching the separation of other tiles
xLabelOffset = -0.18;   % negative => move down (for XLabel)
yLabelOffset = -0.21;   % negative => move left (for YLabel)

% Map 2x2 cells to individual tiles (top-left, top-right, bottom-left, bottom-right)
tileMap = [7, 8; 15, 16];   % top-left, top-right, bottom-left, bottom-right

% Loop cells and display counts with matching label formatting & colors
for r = 1:2
    for c = 1:2
        tileIndex = tileMap(r,c);
        axCell = axesHandles(tileIndex);

        cla(axCell);                      % clear placeholders but keep axes
        axis(axCell, 'square');
        axCell.XLim = [0 1];
        axCell.YLim = [0 1];
        axCell.XTick = [];
        axCell.YTick = [];
        axCell.Box = 'on';

        % select bg color: tiles 7 & 16 blue; 8 & 15 red
        if tileIndex == 7 || tileIndex == 16
            bgColor = blueColor;
        else
            bgColor = redColor;
        end
        axCell.Color = bgColor;

        % Big count in center (white for contrast)
        text(0.5, 0.60, sprintf('%d', confMatTest1(r,c)), ...
            'Parent', axCell, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 28, ...
            'FontWeight', 'bold', ...
            'Color', 'w');

        % Place class labels following the same rules as the histograms:
        lblTrue = char(cmClassNames(r));
        lblPred = char(cmClassNames(c));
        if c == 1
            % left column -> show TRUE class as ylabel (black for contrast)
            ylabel(axCell, lblTrue, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
            % move ylabel outward (normalized units)
            axCell.YLabel.Units = 'normalized';
            % center vertically and push left
            axCell.YLabel.Position = [yLabelOffset, 0.5, 0];
        end
        if r == 2
            % bottom row -> show PRED class as xlabel (black)
            xlabel(axCell, lblPred, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
            % move xlabel outward (normalized units)
            axCell.XLabel.Units = 'normalized';
            axCell.XLabel.Position = [0.5, xLabelOffset, 0];
        end
    end
end


% --- Global labels for the big layout ---
xlabel(tBig, 'Predicted Class', 'FontSize', 14, 'FontWeight', 'bold');
ylabel(tBig, 'Actual Class', 'FontSize', 14, 'FontWeight', 'bold');
title(tBig, 'Confusion Matrix Cell Distributions (All Models)', 'FontSize', 16);

% Save the figure:
savefig(bigFig, fullfile(savePath, char(json.outputFileNames.confusionMatrixGrouped)));



%% EXAMPLE CONFUSION CHART
trueLabels = string(T_ResultsVariable);
ch_Label = string(Label);
ch_wtLabel = string(wtLabel);
ch_OSLabels = string(OSLabels);
ch_trainingDatasetPredictions = string(test1Labels);


figure('Units', 'normalized', 'Position', [0.1 0.3 0.8 0.4]);
sgtitle('Confusion Matrices per model');

subplot(1,4,1);
confusionchart(trueLabels, ch_Label);
title('Standard 5-fold');

subplot(1,4,2);
confusionchart(trueLabels, ch_wtLabel);
title('Weighted 5-fold');

subplot(1,4,3);
confusionchart(trueLabels, ch_OSLabels);
title('Oversampled 5-fold');

subplot(1,4,4);
confusionchart(trueLabels, ch_trainingDatasetPredictions);
title('Final Model');

savefig(fullfile(savePath, char(json.outputFileNames.confusionCharts)));



%% ROC CURVES

    % Variables for BEST performing iterations
    Bfp1  = double(Results(1).max.fp);  % Standard
    Btp1  = double(Results(1).max.tp); 
    Bauc1 = double(Results(1).max.auc);
    Bfp2  = double(Results(2).max.fp);  % Weighted
    Btp2  = double(Results(2).max.tp);  
    Bauc2 = double(Results(2).max.auc);
    Bfp3  = double(Results(3).max.fp);   % Oversampled
    Btp3  = double(Results(3).max.tp);  
    Bauc3 = double(Results(3).max.auc);

    % Variables for WORST performing iterations
    Wfp1  = double(Results(1).min.fp);  % Standard
    Wtp1  = double(Results(1).min.tp); 
    Wauc1 = double(Results(1).min.auc);
    Wfp2  = double(Results(2).min.fp);  % Weighted
    Wtp2  = double(Results(2).min.tp);  
    Wauc2 = double(Results(2).min.auc);
    Wfp3  = double(Results(3).min.fp);   % Oversampled
    Wtp3  = double(Results(3).min.tp);  
    Wauc3 = double(Results(3).min.auc);

    % Variables for CLOSEST TO MEDIAN performing iterations
    Mfp1  = double(Results(1).median_closest.fp);  % Standard
    Mtp1  = double(Results(1).median_closest.tp); 
    Mauc1 = double(Results(1).median_closest.auc);
    Mfp2  = double(Results(2).median_closest.fp);  % Weighted
    Mtp2  = double(Results(2).median_closest.tp);  
    Mauc2 = double(Results(2).median_closest.auc);
    Mfp3  = double(Results(3).median_closest.fp);   % Oversampled
    Mtp3  = double(Results(3).median_closest.tp);  
    Mauc3 = double(Results(3).median_closest.auc);


ROCfig = figure ('Position', [3,393,1676,455]);

    % ROC curve for BEST performing cross-validation models
    subplot(1,3,1); hold on;
        plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');  % Diagonal
        plot(Bfp1,Btp1, 'LineWidth', 2); % Standard
        plot(Bfp2,Btp2, 'LineWidth', 2); % Weighted
        plot(Bfp3,Btp3, 'LineWidth', 2); % Oversampled
        plot(fp4,tp4, 'LineWidth', 2);   % Test1

    legend({ ...
        sprintf('cvStandard (AUC = %.3f)', Bauc1), ...
        sprintf('cvWeighted (AUC = %.3f)', Bauc2), ...
        sprintf('cvOversampled (AUC = %.3f)', Bauc3), ...
        sprintf('TrainingData (AUC = %.3f)', aucTest1)}, ...
        'Location', 'SouthEast');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('BEST performing iterations');
    grid on;

    % ROC curve for WORST performing cross-validation models
    subplot(1,3,2); hold on;
        plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');  % Diagonal
        plot(Wfp1,Wtp1, 'LineWidth', 2); % Standard
        plot(Wfp2,Wtp2, 'LineWidth', 2); % Weighted
        plot(Wfp3,Wtp3, 'LineWidth', 2); % Oversampled
        plot(fp4,tp4, 'LineWidth', 2);   % Test1

    legend({ ...
        sprintf('cvStandard (AUC = %.3f)', Wauc1), ...
        sprintf('cvWeighted (AUC = %.3f)', Wauc2), ...
        sprintf('cvOversampled (AUC = %.3f)', Wauc3), ...
        sprintf('TrainingData (AUC = %.3f)', aucTest1)}, ...
        'Location', 'SouthEast');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('WORST performing iterations');
    grid on;

    % ROC curve for MEDIAN value of cross-validation models
    subplot(1,3,3); hold on;
        plot([0 1], [0 1], 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');  % Diagonal
        plot(Mfp1,Mtp1, 'LineWidth', 2); % Standard
        plot(Mfp2,Mtp2, 'LineWidth', 2); % Weighted
        plot(Mfp3,Mtp3, 'LineWidth', 2); % Oversampled
        plot(fp4,tp4, 'LineWidth', 2);   % Test1

    legend({ ...
        sprintf('cvStandard (AUC = %.3f)', Mauc1), ...
        sprintf('cvWeighted (AUC = %.3f)', Mauc2), ...
        sprintf('cvOversampled (AUC = %.3f)', Mauc3), ...
        sprintf('TrainingData (AUC = %.3f)', aucTest1)}, ...
        'Location', 'SouthEast');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('CLOSEST to MEDIAN performing iterations');
    grid on;

% --- Global labels for the big layout ---
sgtitle('ROC Curves for All Models', 'FontSize', 16);

% Save the figure:
savefig(ROCfig, fullfile(savePath, char(json.outputFileNames.ROC_Curves)));



%% CREATE AN EXCEL FILE 
excelFileName = fullfile(savePath, char(json.outputFileNames.excelFileName));

% Write error summary to sheet1
writetable(ErrorTable, excelFileName, 'Sheet', 'Errors', 'WriteRowNames', true);

% Write predictor importance to sheet2
writetable(PredictorImportanceTable, excelFileName, 'Sheet', 'PredictorImportance');