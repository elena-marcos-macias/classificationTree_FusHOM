%% =========================================================================
%  DECISION TREE CLASSIFIER WITH REPEATED K-FOLD CROSS-VALIDATION
%  =========================================================================
%  Trains a CART classification tree on all available data (full model),
%  then evaluates generalisation performance via repeated stratified k-fold
%  cross-validation. Exports predictor importance, per-fold metrics, ROC
%  curves, histograms, and optionally evaluates an independent test set.
%
%  KEY DESIGN DECISIONS
%  --------------------
%  · Full model  : trained on 100 % of the data — used ONLY for tree
%                  visualisation and global predictor importance. Its in-
%                  sample error is optimistically biased and must not be
%                  used as a performance estimate.
%  · CV models   : each run calls fitctree with 'KFold', which builds k
%                  trees internally. Tree f is trained on all folds except
%                  fold f ( ≈ (k-1)/k of the data). kfoldPredict / kfold-
%                  Loss aggregate the held-out predictions to give an
%                  unbiased error estimate for the whole run.
%  · Positive class: read from JSON so the orientation of ROC / AUC is
%                  explicit and reproducible, regardless of class balance.
%
%  INPUTS  (configured via data/instructionsDecisionTree.json)
%  -------
%    trainingDataFile  – Excel file with predictors + response column
%    testDataFile      – Independent Excel file (same column structure)
%    positiveClass     – String label of the class of interest (e.g. "SUD")
%    catVariable       – Name of the single categorical predictor column
%    resultsVariable   – Name of the response (target) column
%    nFolds / nRuns    – Cross-validation grid
%    outputFileNames   – Names for the exported tree PDF and Excel file
%
%  OUTPUTS (saved to ./results/)
%  -------
%    DecisionTree.jpg        – Full-model tree graph (300 dpi raster)
%    <excelFileName>.xlsx
%      · Errors              – Mean, SD, best, worst, median CV metrics
%      · PredictorImportance – Gini importance from the full model
%      · CVTreeImportance    – Importance from every CV training tree
%      · CVTreeSummary       – Aggregate stats across all CV trees
%      · AUC_CI              – Mean AUC and 95 % CI from repeated CV
%      · AUC_CV_Distribution – Raw AUC values across all runs
%    Histograms.jpg          – Error-rate and AUC distributions across runs
%    ROCcurves.jpg           – Best / median / mean / worst ROC curves
%    AUC_CV_Distribution.jpg – Bootstrap-style AUC distribution with 95 % CI
%
%  DEPENDENCIES
%  ------------
%    Statistics and Machine Learning Toolbox (fitctree, perfcurve, etc.)
%    selectColumns()  – custom helper in ./utils/
%
%  Author : [Elena Marcos-Macías]
%  Date   : [19/05/2026]
% =========================================================================


%% ---- 1. PATH & OUTPUT FOLDER SETUP -------------------------------------
addpath(genpath('./data'));
addpath(genpath('./utils'));
addpath('./requirements');

savePath = './results/';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% --- Toolbox check --------------------------------------------------------
% The Statistics and Machine Learning Toolbox is required. Fail early with
% a clear message rather than crashing silently inside fitctree.
requiredToolbox = 'Statistics and Machine Learning Toolbox';
installedToolboxes = {ver().Name};
if ~any(strcmp(installedToolboxes, requiredToolbox))
    error(['Required toolbox not found: "%s".\n' ...
           'Install it via the MATLAB Add-On Explorer before running this script.'], ...
           requiredToolbox);
end

% --- Utility function check -----------------------------------------------
if ~exist('selectColumns', 'file')
    error(['selectColumns() not found on the MATLAB path.\n' ...
           'Ensure the file exists in ./utils/ and that addpath above succeeded.\n' ...
           'Current path entry for utils: %s'], fullfile(pwd, 'utils'));
end


%% ---- 2. LOAD CONFIGURATION ---------------------------------------------
jsonPath = 'data/instructionsDecisionTree.json';

% --- JSON file existence check --------------------------------------------
if ~isfile(jsonPath)
    error(['Configuration file not found: "%s".\n' ...
           'Expected location: %s'], jsonPath, fullfile(pwd, jsonPath));
end

json = readstruct(jsonPath);

% --- Required JSON fields check -------------------------------------------
% Verify that all top-level blocks and critical sub-fields exist before
% any downstream code attempts to index into them.
requiredFields = { ...
    'inputDataSelection', ...
    'crossValidationMethods', ...
    'outputFileNames'};

for i = 1:numel(requiredFields)
    if ~isfield(json, requiredFields{i})
        error('Missing required field "%s" in JSON configuration file.', requiredFields{i});
    end
end

requiredInputFields = {'trainingDataFile','testDataFile','positiveClass', ...
                       'catVariable','resultsVariable','columnCriteria'};
for i = 1:numel(requiredInputFields)
    if ~isfield(json.inputDataSelection, requiredInputFields{i})
        error('Missing required field "inputDataSelection.%s" in JSON.', requiredInputFields{i});
    end
end

if ~isfield(json.inputDataSelection.columnCriteria, 'target_columns') || ...
   ~isfield(json.inputDataSelection.columnCriteria, 'ignore_columns')
    error('JSON field "inputDataSelection.columnCriteria" must contain "target_columns" and "ignore_columns".');
end

if ~isfield(json.crossValidationMethods, 'nFolds') || ...
   ~isfield(json.crossValidationMethods, 'nRuns')
    error('JSON field "crossValidationMethods" must contain "nFolds" and "nRuns".');
end

if ~isfield(json.outputFileNames, 'excelFileName') || ...
   ~isfield(json.outputFileNames, 'testExcelFile')
    error('JSON field "outputFileNames" must contain "excelFileName" and "testExcelFile".');
end


%% ---- 3. LOAD & PREPARE TRAINING DATA -----------------------------------
% Read the training Excel file; drop any row that contains a missing value
% (listwise deletion keeps the design matrix rectangular and consistent).
trainingDataFile = char(json.inputDataSelection.trainingDataFile);
trainingDataPath = fullfile('./data', trainingDataFile);

% --- Training file existence check ----------------------------------------
if ~isfile(trainingDataPath)
    error(['Training data file not found: "%s".\n' ...
           'Check the "trainingDataFile" field in the JSON and ensure the file\n' ...
           'is located in the ./data/ folder.'], trainingDataPath);
end

T_Raw      = readtable(trainingDataPath);
nMissing   = sum(any(ismissing(T_Raw), 2));
T_Original = rmmissing(T_Raw);

if nMissing > 0
    warning('%d row(s) with missing values removed from the training set.', nMissing);
end

% --- Empty table check ----------------------------------------------------
if isempty(T_Original)
    error(['Training dataset is empty after removing missing values.\n' ...
           'Check the file "%s" for data integrity.'], trainingDataPath);
end

% --- Column selection & response variable check ---------------------------
T_Data = selectColumns(T_Original, ...
    json.inputDataSelection.columnCriteria.target_columns, ...
    json.inputDataSelection.columnCriteria.ignore_columns);

if isempty(T_Data)
    error(['selectColumns() returned an empty table.\n' ...
           'Verify that "target_columns" and "ignore_columns" in the JSON\n' ...
           'produce at least one predictor column.']);
end

resultsVarName = json.inputDataSelection.resultsVariable;
if ~ismember(resultsVarName, T_Original.Properties.VariableNames)
    error(['Response variable "%s" not found in the training data.\n' ...
           'Available columns: %s'], ...
           resultsVarName, strjoin(T_Original.Properties.VariableNames, ', '));
end
T_ResultsVariable = T_Original.(resultsVarName);

catVariable = char(json.inputDataSelection.catVariable);
if ~ismember(catVariable, T_Data.Properties.VariableNames)
    error(['Categorical predictor "%s" not found in the selected predictor columns.\n' ...
           'Available predictors: %s'], ...
           catVariable, strjoin(T_Data.Properties.VariableNames, ', '));
end

nObs = height(T_Data);

% --- Positive class validation --------------------------------------------
% The positive class is declared explicitly in the JSON instead of being
% inferred automatically (e.g. as the minority class), so that ROC
% orientation is deterministic across runs and data subsets.
positiveClass = string(json.inputDataSelection.positiveClass);
classNames    = unique(string(T_ResultsVariable));

if numel(classNames) < 2
    error(['Only one class found in the response variable ("%s").\n' ...
           'A binary or multi-class response with at least 2 classes is required.'], ...
           classNames(1));
end

if ~any(classNames == positiveClass)
    error(['Positive class "%s" not found in the response variable.\n' ...
           'Available classes: %s'], positiveClass, strjoin(classNames, ', '));
end

negativeClasses = classNames(classNames ~= positiveClass);
fprintf('Positive class : %s\n',   positiveClass);
fprintf('Negative class : %s\n',   strjoin(negativeClasses, ', '));
fprintf('Total samples  : %d\n\n', nObs);

% Binary ground-truth vector used for perfcurve (true = positive class)
trueLabels = string(T_ResultsVariable);
trueBinary = (trueLabels == positiveClass);

% --- Class balance warning ------------------------------------------------
posCount = sum(trueBinary);
negCount = nObs - posCount;
minorRatio = min(posCount, negCount) / nObs;
if minorRatio < 0.15
    warning(['Severe class imbalance detected: %d positive vs %d negative samples (%.1f %% minority).\n' ...
             'Consider oversampling, undersampling, or cost-sensitive learning.'], ...
             posCount, negCount, 100 * minorRatio);
end


%% ---- 4. FULL-MODEL DECISION TREE (visualisation + global importance) ---
Mdl = fitctree(T_Data, T_ResultsVariable, ...
    'CategoricalPredictors', {catVariable}, ...
    'MinParentSize', 3);

% --- Draw tree into a standard figure (required for exportgraphics) ------
% view(Mdl, figHandle, 'Mode', 'graph') renders the tree inside a normal
% MATLAB figure instead of the interactive app viewer, allowing raster
% export at arbitrary resolution.
treeFigRaster = figure('Visible', 'off', ...   % suppress screen flash
    'Units',    'pixels', ...
    'Position', [100 100 1800 1000]);           % wide canvas for deep trees

view(Mdl, treeFigRaster, 'Mode', 'graph');
drawnow;

% --- Increase font size on all text elements in the tree -----------------
% view() populates the figure with text objects for node labels, split
% rules, and class names. Find them all and scale the font up uniformly.
allText = findall(treeFigRaster, 'Type', 'text');
set(allText, 'FontSize', 18);          % increase as desired (default ≈ 8–9)
drawnow;                               % reflow layout after font change

% --- Export as JPG only --------------------------------------------------
treeJpgPath = fullfile(savePath, 'DecisionTree.jpg');
exportgraphics(treeFigRaster, treeJpgPath, 'Resolution', 300);

% --- Export verification --------------------------------------------------
if ~isfile(treeJpgPath)
    warning('DecisionTree.jpg was not found after export — check disk space and write permissions.');
else
    fprintf('Tree exported: DecisionTree.jpg\n');
end
close(treeFigRaster);

% --- Predictor importance bar chart (full model) -------------------------
imp            = predictorImportance(Mdl);
predictorNames = Mdl.PredictorNames;

figure;
bar(imp);
title('Predictor Importance (Full Model — all data)');
xticks(1:numel(predictorNames));
xticklabels(predictorNames);
xtickangle(45);
ylabel('Importance Score');

PredictorImportanceTable = table(predictorNames', imp', ...
    'VariableNames', {'Predictor', 'Importance'});


%% ---- 5. CROSS-VALIDATION SETUP -----------------------------------------
nFolds = json.crossValidationMethods.nFolds;
nRuns  = json.crossValidationMethods.nRuns;

% --- CV parameter validation ----------------------------------------------
if ~isnumeric(nFolds) || ~isscalar(nFolds) || nFolds < 2 || nFolds ~= floor(nFolds)
    error('"nFolds" must be a positive integer >= 2. Received: %s', mat2str(nFolds));
end

if ~isnumeric(nRuns) || ~isscalar(nRuns) || nRuns < 1 || nRuns ~= floor(nRuns)
    error('"nRuns" must be a positive integer >= 1. Received: %s', mat2str(nRuns));
end

if nFolds > nObs
    error(['"nFolds" (%d) cannot exceed the number of observations (%d).\n' ...
           'Reduce nFolds or provide more data.'], nFolds, nObs);
end

% Approximate train / held-out split per fold (may differ by ±1 sample
% when nObs is not evenly divisible by nFolds)
nTrain_approx = round(nObs * (nFolds - 1) / nFolds);
nTest_approx  = nObs - nTrain_approx;

fprintf('CV setup : %d folds × %d runs\n', nFolds, nRuns);
fprintf('Per tree : ~%d training | ~%d held-out (%.1f %% / %.1f %%)\n\n', ...
    nTrain_approx, nTest_approx, ...
    100 * nTrain_approx / nObs, 100 * nTest_approx / nObs);

% --- Preallocate storage --------------------------------------------------
nTotalTrees = nRuns * nFolds;   % one CV training tree per fold per run
allPredictors = Mdl.PredictorNames;
nPredictors   = numel(allPredictors);

errorResults        = zeros(nRuns, 2);           % [errorRate, AUC] per run
FPAll               = cell(nRuns, 1);            % FPR vectors for ROC
TPAll               = cell(nRuns, 1);            % TPR vectors for ROC
AUCs                = zeros(nRuns, 1);           % scalar AUC per run

CVTreeImportanceRaw = zeros(nTotalTrees, nPredictors); % importance matrix
CVTreeRunID         = zeros(nTotalTrees, 1);
CVTreeFoldID        = zeros(nTotalTrees, 1);
CVTreeTrainSize     = zeros(nTotalTrees, 1);
CVTreeTestSize      = zeros(nTotalTrees, 1);

treeCounter = 0;


%% ---- 6. REPEATED K-FOLD CROSS-VALIDATION LOOP --------------------------
for run = 1:nRuns
    fprintf('Run %d / %d\n', run, nRuns);

    % fitctree with 'KFold' internally builds nFolds trees, each trained
    % on (k-1) folds and tested on the remaining fold. It does NOT train a
    % single model — CVMdl is a container of k training trees.
    % Class stratification is intentionally NOT enforced so that fold
    % composition reflects the natural class distribution of the data,
    % which may differ from 50/50 in real experimental conditions.
    CVMdl = fitctree(T_Data, T_ResultsVariable, ...
        'KFold',                nFolds, ...
        'CategoricalPredictors',{catVariable}, ...
        'MinParentSize',        3);

    % --- Run-level performance (aggregated over all folds) ----------------
    [~, Score]    = kfoldPredict(CVMdl);  % out-of-fold predicted scores
    missClassRate = kfoldLoss(CVMdl);     % out-of-fold error rate

    % --- Score matrix sanity check ----------------------------------------
    if size(Score, 2) < 2
        error(['kfoldPredict returned a score matrix with fewer than 2 columns (run %d).\n' ...
               'This may indicate a degenerate fold where only one class is present.\n' ...
               'Try increasing nFolds or the dataset size.'], run);
    end

    % Locate the positive class column in the score matrix.
    % Computed once per run because CVMdl.ClassNames is consistent within a
    % run (MATLAB preserves the training-data class order).
    cvClassNames = string(CVMdl.ClassNames);
    posClassIdx  = find(cvClassNames == positiveClass, 1);
    if isempty(posClassIdx)
        error(['Positive class "%s" not found in CVMdl.ClassNames (run %d).\n' ...
               'Verify the JSON label matches the data exactly.\n' ...
               'Classes found: %s'], ...
               positiveClass, run, strjoin(cvClassNames, ', '));
    end

    % --- Degenerate run check ---------------------------------------------
    % Without enforced stratification, occasional runs may produce a fold
    % where all held-out samples belong to the same class. perfcurve would
    % still compute a value but the AUC would be uninformative. Detect this
    % condition and skip the run rather than letting it silently bias the
    % results.
    predictedLabels = string(kfoldPredict(CVMdl));
    if numel(unique(predictedLabels)) < 2
        warning(['Run %d skipped: out-of-fold predictions contain only one class.\n' ...
                 'This can occur without stratification when class distribution\n' ...
                 'is unequal. Consider increasing nFolds or nRuns to compensate.'], run);
        continue;
    end

    [fp, tp, ~, auc] = perfcurve(trueBinary, Score(:, posClassIdx), true);

    % --- AUC range check --------------------------------------------------
    if auc < 0 || auc > 1
        warning('Unexpected AUC value %.4f in run %d. Expected range [0, 1].', auc, run);
    end

    errorResults(run, :) = [missClassRate, auc];
    FPAll{run} = fp(:);
    TPAll{run} = tp(:);
    AUCs(run)  = auc;

    % --- Per-CV-tree predictor importance ---------------------------------
    % CVMdl.Trained{f} is the tree trained on all folds except fold f.
    % Its predictor order is guaranteed identical to allPredictors, so the
    % importance vector maps directly to the columns of CVTreeImportanceRaw.
    for fold = 1:nFolds
        treeCounter = treeCounter + 1;

        cvTree    = CVMdl.Trained{fold};
        partition = CVMdl.Partition;

        % --- CV tree integrity check --------------------------------------
        if isempty(cvTree)
            error('CVMdl.Trained{%d} is empty in run %d. Training may have failed.', fold, run);
        end

        % Derive actual (not approximate) train / test counts from the
        % partition object so per-tree metadata is exact.
        testIdx      = test(partition, fold);
        nTestActual  = sum(testIdx);
        nTrainActual = nObs - nTestActual;

        % --- Fold size check ----------------------------------------------
        if nTestActual == 0
            error('Fold %d in run %d has zero test samples. Check data size vs nFolds.', fold, run);
        end

        CVTreeImportanceRaw(treeCounter, :) = predictorImportance(cvTree);
        CVTreeRunID(treeCounter)            = run;
        CVTreeFoldID(treeCounter)           = fold;
        CVTreeTrainSize(treeCounter)        = nTrainActual;
        CVTreeTestSize(treeCounter)         = nTestActual;
    end
end

% --- Post-loop CV results check -------------------------------------------
% Remove preallocated zero rows corresponding to skipped runs before
% computing summary statistics. A run is considered skipped if its AUC
% was never written (remains 0 AND errorResults row is also 0).
skippedRuns = (AUCs == 0) & (errorResults(:,1) == 0);
if any(skippedRuns)
    warning('%d run(s) were skipped due to degenerate folds and excluded from results.', ...
            sum(skippedRuns));
    AUCs         = AUCs(~skippedRuns);
    errorResults = errorResults(~skippedRuns, :);
    FPAll        = FPAll(~skippedRuns);
    TPAll        = TPAll(~skippedRuns);
end

if isempty(AUCs)
    error(['All CV runs were skipped. No valid AUC values available.\n' ...
           'Check class distribution and consider increasing nFolds.']);
end

if all(AUCs == 0)
    warning(['All AUC values are 0. This is very unusual.\n' ...
             'Check that the positive class label in the JSON ("%s") exactly\n' ...
             'matches the values in the response column (case-sensitive).'], positiveClass);
end

% Update nRuns to reflect the actual number of valid runs used
nRuns = numel(AUCs);
fprintf('\nValid runs used for analysis: %d\n', nRuns);



%% ---- 7. CV-TREE IMPORTANCE TABLES ---------------------------------------

% --- Full detail table: one row per CV training tree ---------------------
CVTreeImportanceTable = array2table( ...
    [CVTreeRunID, CVTreeFoldID, CVTreeTrainSize, CVTreeTestSize, CVTreeImportanceRaw], ...
    'VariableNames', [{'Run','Fold','TrainSize','TestSize'}, allPredictors]);

% --- Summary statistics across all nRuns × nFolds training trees ---------
meanImpCV = mean(CVTreeImportanceRaw, 1);
stdImpCV  = std(CVTreeImportanceRaw,  0, 1);
medImpCV  = median(CVTreeImportanceRaw, 1);
usageRate = mean(CVTreeImportanceRaw > 0, 1);

minImpCV = min(CVTreeImportanceRaw, [], 1);
maxImpCV = max(CVTreeImportanceRaw, [], 1);

valuesRangeCell = arrayfun( ...
    @(lo, hi) sprintf('[%.3f, %.3f]', lo, hi), ...
    minImpCV, maxImpCV, ...
    'UniformOutput', false);

% --- Numeric summary table (Mean / SD / Median / UsageRate) --------------
CVTreeSummaryTable = array2table([meanImpCV; stdImpCV; medImpCV; usageRate], ...
    'VariableNames', allPredictors, ...
    'RowNames', {'Mean', 'SD', 'Median', 'UsageRate'});

% --- Cell row with the range of values for each predictor ----------------
% This row will be written below the numeric summary in Excel.
CVTreeSummaryValuesRange = [{'ValuesRange'}, valuesRangeCell];

disp('CVTreeSummaryTable:');
disp(CVTreeSummaryTable);


%% ---- 8. SELECT REPRESENTATIVE RUNS (best / worst / median / mean) -------
% Representative runs are chosen by AUC so that ROC curves are anchored to
% real observed data rather than a synthetic average (except the mean curve,
% which is computed by interpolation in Section 10).

[maxAuc, idxMax] = max(AUCs);
[minAuc, idxMin] = min(AUCs);

medAuc       = median(AUCs);
[~, idxMed]  = min(abs(AUCs - medAuc));   % run whose AUC is closest to the median

meanAuc      = mean(AUCs);
[~, idxMean] = min(abs(AUCs - meanAuc));  % run whose AUC is closest to the mean

% Package into a struct for tidy downstream access
Results.max.fp  = FPAll{idxMax};   Results.max.tp  = TPAll{idxMax};   Results.max.auc  = maxAuc;
Results.min.fp  = FPAll{idxMin};   Results.min.tp  = TPAll{idxMin};   Results.min.auc  = minAuc;
Results.med.fp  = FPAll{idxMed};   Results.med.tp  = TPAll{idxMed};   Results.med.auc  = AUCs(idxMed);
Results.mean.fp = FPAll{idxMean};  Results.mean.tp = TPAll{idxMean};  Results.mean.auc = AUCs(idxMean);


%% ---- 9. CV SUMMARY TABLE (error rate & AUC across runs) ----------------
errorCol = [mean(errorResults(:,1));
            std(errorResults(:,1));
            errorResults(idxMax, 1);   % error of the best-AUC run
            errorResults(idxMin, 1);   % error of the worst-AUC run
            errorResults(idxMed, 1)];  % error of the median-closest run

aucCol = [mean(AUCs);
          std(AUCs);
          AUCs(idxMax);
          AUCs(idxMin);
          AUCs(idxMed)];

ErrorTable = table(errorCol, aucCol, ...
    'VariableNames', {'Error','AUC'}, ...
    'RowNames',      {'MeanCV','StdCV','Best','Worst','MedianClosest'});


%% ---- 10. EXPORT TO EXCEL ------------------------------------------------
excelFileName = fullfile(savePath, char(json.outputFileNames.excelFileName));

% --- Excel write with verification ----------------------------------------
try
    writetable(ErrorTable,               excelFileName, 'Sheet','Errors',             'WriteRowNames', true);
    writetable(PredictorImportanceTable, excelFileName, 'Sheet','PredictorImportance');
    writetable(CVTreeImportanceTable,    excelFileName, 'Sheet','CVTreeImportance');
    writetable(CVTreeSummaryTable,       excelFileName, 'Sheet','CVTreeSummary',       'WriteRowNames', true);
    writecell(CVTreeSummaryValuesRange,  excelFileName, 'Sheet','CVTreeSummary',       'Range','A6');
catch ME
    error(['Failed to write Excel file "%s".\n' ...
           'Possible causes: file is open in Excel, insufficient disk space, or bad path.\n' ...
           'MATLAB error: %s'], excelFileName, ME.message);
end

if ~isfile(excelFileName)
    warning('Excel file not found after write attempt: %s', excelFileName);
else
    fprintf('Results exported to: %s\n', excelFileName);
end


%% ---- 11. HISTOGRAMS (error rate & AUC distributions) -------------------
figure;

% Helper to add padding and aligned tick marks to a histogram axes
function styleHistogram(h, ax)
    binW = h.BinWidth;
    ylim(ax, [0, max(h.Values) * 1.15]);
    xlim(ax, [min(h.BinEdges) - 0.5*binW, max(h.BinEdges) + 0.5*binW]);
    xticks(ax, h.BinEdges);
end

meanErrorRate = mean(errorResults(:,1));
meanAUC       = mean(errorResults(:,2));

ax1 = subplot(1, 2, 1);
h1  = histogram(ax1, errorResults(:,1), 'NumBins', 10);
title(ax1, 'ErrorRate Distribution');
grid(ax1, 'on'); pbaspect(ax1, [1 1 1]);
styleHistogram(h1, ax1);
hold(ax1, 'on');
xline(ax1, meanErrorRate, 'r-', 'LineWidth', 1.5, ...
    'Label', sprintf('%.3f', meanErrorRate), ...
    'FontSize', 7, ...
    'LabelVerticalAlignment', 'middle', ...
    'LabelHorizontalAlignment', 'center');
hold(ax1, 'off');

ax2 = subplot(1, 2, 2);
h2  = histogram(ax2, errorResults(:,2), 'NumBins', 10);
title(ax2, 'AUC Distribution');
grid(ax2, 'on'); pbaspect(ax2, [1 1 1]);
styleHistogram(h2, ax2);
hold(ax2, 'on');
xline(ax2, meanAUC, 'r-', 'LineWidth', 1.5, ...
    'Label', sprintf('%.3f', meanAUC), ...
    'FontSize', 7, ...
    'LabelVerticalAlignment', 'middle', ...
    'LabelHorizontalAlignment', 'center');
hold(ax2, 'off');

histPath = fullfile(savePath, 'Histograms.jpg');
exportgraphics(gcf, histPath, 'Resolution', 300);
if ~isfile(histPath)
    warning('Histograms.jpg was not found after export.');
end


%% ---- 12. ROC CURVES -----------------------------------------------------
% Build a mean ROC curve by interpolating all per-run curves onto a shared
% FPR grid. Steps:
%   1. Anchor each curve at (0,0) and (1,1) to prevent extrapolation.
%   2. Remove duplicate FPR values (unique + 'sorted') — required by interp1.
%   3. Clamp interpolated TPR to [0, 1] to handle floating-point edge cases.
fprGrid    = linspace(0, 1, 200);
tprGridAll = nan(numel(fprGrid), nRuns);

for run = 1:nRuns
    fp = [0; FPAll{run}(:); 1];    % anchor at both ends
    tp = [0; TPAll{run}(:); 1];

    [fpUniq, ia] = unique(fp, 'sorted');  % deduplicate for interp1
    tpUniq       = tp(ia);

    % --- Interpolation input check ----------------------------------------
    if numel(fpUniq) < 2
        warning('Run %d: insufficient unique FPR points for interpolation. ROC curve may be unreliable.', run);
        tprGridAll(:, run) = zeros(numel(fprGrid), 1);
        continue;
    end

    tprGridAll(:, run) = max(0, min(1, interp1(fpUniq, tpUniq, fprGrid, 'linear')));
end

meanTP = mean(tprGridAll, 2);   % element-wise mean across all runs

colors = parula(10);

figure; hold on; grid on;
plot(Results.max.fp,  Results.max.tp,  '-',  'LineWidth', 2, 'Color', colors(1,:));
plot(Results.med.fp,  Results.med.tp,  '--', 'LineWidth', 2, 'Color', colors(4,:));
plot(fprGrid,         meanTP,          '-',  'LineWidth', 2, 'Color', colors(7,:));
plot(Results.min.fp,  Results.min.tp,  '-',  'LineWidth', 2, 'Color', colors(8,:));
plot([0 1],           [0 1],           'k--','LineWidth', 0.5);

legend({ ...
    sprintf('Best           (AUC = %.3f)', Results.max.auc), ...
    sprintf('Median-closest (AUC = %.3f)', Results.med.auc), ...
    sprintf('Mean           (AUC = %.3f)', meanAuc), ...
    sprintf('Worst          (AUC = %.3f)', Results.min.auc), ...
    'Random'}, 'Location', 'SouthEast');

xlabel('FPR');
ylabel('TPR');
title('ROC Curves — Repeated CV');

rocPath = fullfile(savePath, 'ROCcurves.jpg');
exportgraphics(gcf, rocPath, 'Resolution', 300);
fprintf('\nAll figures and tables saved to: %s\n', savePath);

%% ---- 13. 95 % CONFIDENCE INTERVAL FOR AUC (from repeated CV runs) ------
% With nRuns repeated k-fold CV already completed in Section 6, the vector
% AUCs contains one empirical AUC per run. Each run uses a different random
% partition, so these values are approximately independent estimates of the
% model's true generalisation AUC.
%
% The 95 % CI is computed as the 2.5th and 97.5th percentiles of that
% distribution — equivalent to a percentile bootstrap where the "resampling"
% is the repeated CV itself. No additional model fitting is required.
%
% This approach is preferred over pooling out-of-fold predictions across
% runs when nRuns is large (e.g. 1000), because pooling inflates the
% effective sample size and produces an artificially narrow interval.

CI_lower = prctile(AUCs, 2.5);
CI_upper = prctile(AUCs, 97.5);
CI_mean  = mean(AUCs);

fprintf('\n===== 95 %% CI FOR AUC (repeated CV) =====\n');
fprintf('Runs used  : %d\n',         nRuns);
fprintf('Mean AUC   : %.4f\n',       CI_mean);
fprintf('95 %% CI   : [%.4f, %.4f]\n\n', CI_lower, CI_upper);

% --- Export to Excel ------------------------------------------------------
CISummaryTable = table([CI_mean; CI_lower; CI_upper], ...
    'VariableNames', {'Value'}, ...
    'RowNames',      {'MeanAUC', 'CI_Lower_2p5', 'CI_Upper_97p5'});

try
    writetable(CISummaryTable, excelFileName, ...
        'Sheet', 'AUC_CI', 'WriteRowNames', true);
catch ME
    error(['Failed to write CI results to Excel file "%s".\n' ...
           'MATLAB error: %s'], excelFileName, ME.message);
end

% --- Plot distribution of AUCs across runs --------------------------------
figure;
histogram(AUCs, 'NumBins', 40, 'Normalization', 'count');
hold on;
xline(CI_lower, 'r--', 'LineWidth', 1.5, ...
    'Label', sprintf('2.5%%  %.3f', CI_lower), ...
    'FontSize', 8, 'LabelVerticalAlignment', 'middle');
xline(CI_upper, 'r--', 'LineWidth', 1.5, ...
    'Label', sprintf('97.5%% %.3f', CI_upper), ...
    'FontSize', 8, 'LabelVerticalAlignment', 'middle');
xline(CI_mean,  'k-',  'LineWidth', 1.5, ...
    'Label', sprintf('Mean  %.3f', CI_mean), ...
    'FontSize', 8, 'LabelVerticalAlignment', 'middle');
hold off;
title('AUC Distribution across Repeated CV Runs (95 % CI)');
xlabel('AUC');
ylabel('Count');
grid on;

AUCDistPath = fullfile(savePath, 'AUC_CV_Distribution.jpg');
exportgraphics(gcf, AUCDistPath, 'Resolution', 300);

if ~isfile(AUCDistPath)
    warning('AUC_CV_Distribution.jpg was not found after export.');
else
    fprintf('AUC distribution figure saved: AUC_CV_Distribution.jpg\n');
end


%% ---- 14. EXTERNAL TEST SET EVALUATION (optional) -----------------------
% This section evaluates the full model on an independent held-out test set
% if one is available. If the file specified in "testDataFile" does not
% exist, the section is skipped with a warning and no error is raised.
%
% NOTE: In the current study no independent test set is available. This
% section is retained for completeness and future use, but will not execute.
testDataFile = char(json.inputDataSelection.testDataFile);
testDataPath = fullfile('./data', testDataFile);

if ~isfile(testDataPath)
    warning(['Test data file not found: "%s".\n' ...
             'Skipping external test set evaluation (Section 14).'], testDataPath);
else

    T_TestRaw      = readtable(testDataPath);
    nMissingTest   = sum(any(ismissing(T_TestRaw), 2));
    T_TestOriginal = rmmissing(T_TestRaw);

    if nMissingTest > 0
        warning('%d row(s) with missing values removed from the test set.', nMissingTest);
    end

    if isempty(T_TestOriginal)
        error(['Test dataset is empty after removing missing values.\n' ...
               'Check the file: %s'], testDataPath);
    end

    % --- Test response variable check -------------------------------------
    if ~ismember(resultsVarName, T_TestOriginal.Properties.VariableNames)
        error(['Response variable "%s" not found in the test data.\n' ...
               'Available columns: %s'], ...
               resultsVarName, strjoin(T_TestOriginal.Properties.VariableNames, ', '));
    end

    % Apply the same column-selection logic used on the training set
    T_TestData            = selectColumns(T_TestOriginal, ...
        json.inputDataSelection.columnCriteria.target_columns, ...
        json.inputDataSelection.columnCriteria.ignore_columns);
    T_TestResultsVariable = T_TestOriginal.(resultsVarName);

    if isempty(T_TestResultsVariable)
        error('Test labels are empty after column selection. Check resultsVariable in JSON or the test file.');
    end

    % --- Test predictor column consistency check --------------------------
    trainPredictors = T_Data.Properties.VariableNames;
    testPredictors  = T_TestData.Properties.VariableNames;
    missingCols     = setdiff(trainPredictors, testPredictors);
    if ~isempty(missingCols)
        error(['Test data is missing predictor column(s) present in training data:\n  %s\n' ...
               'Ensure both files share the same column structure.'], ...
               strjoin(missingCols, ', '));
    end

    testLabelsTrue = string(T_TestResultsVariable);
    testBinary     = (testLabelsTrue == positiveClass);

    % --- Test set class presence warning ----------------------------------
    testClassNames = unique(testLabelsTrue);
    if ~any(testClassNames == positiveClass)
        warning(['Positive class "%s" does not appear in the test set.\n' ...
                 'AUC will be undefined. Test classes found: %s'], ...
                 positiveClass, strjoin(testClassNames, ', '));
    end

    fprintf('\n===== TEST SET CLASS DISTRIBUTION =====\n');
    disp(tabulate(testLabelsTrue));

    % --- Predict with the full model --------------------------------------
    [testPredLabels, testScores] = predict(Mdl, T_TestData);
    testPredLabels = string(testPredLabels);

    % --- Score matrix check -----------------------------------------------
    if isempty(testScores) || size(testScores, 1) ~= height(T_TestData)
        error('predict() returned an unexpected score matrix size for the test set.');
    end

    % --- Error rate -------------------------------------------------------
    testErrorRate = mean(testPredLabels ~= testLabelsTrue);

    % --- AUC --------------------------------------------------------------
    % Guard against a degenerate test set that contains only one class
    % (perfcurve would crash; AUC is mathematically undefined in that case).
    if numel(unique(testBinary)) < 2
        warning('Test set contains only one class — AUC is undefined.');
        fpTest = []; tpTest = []; aucTest = NaN;
    else
        mdlClassNames = string(Mdl.ClassNames);
        posClassIdx   = find(mdlClassNames == positiveClass, 1);
        if isempty(posClassIdx)
            error('Positive class "%s" not found in Mdl.ClassNames.', positiveClass);
        end
        [fpTest, tpTest, ~, aucTest] = perfcurve(testBinary, testScores(:, posClassIdx), true);
    end

    % --- Report results ---------------------------------------------------
    fprintf('\n===== EXTERNAL TEST SET RESULTS =====\n');
    fprintf('Test samples : %d\n',   numel(testLabelsTrue));
    fprintf('Error rate   : %.4f\n', testErrorRate);
    if isnan(aucTest)
        fprintf('AUC          : undefined (single class in test set)\n\n');
    else
        fprintf('AUC          : %.4f\n\n', aucTest);
    end

    % --- Confusion matrix -------------------------------------------------
    figure;
    confusionchart(testLabelsTrue, testPredLabels);
    title('External Test Set — Confusion Matrix');
    exportgraphics(gcf, fullfile(savePath, 'Confusion_TestSet.jpg'), 'Resolution', 300);

    % --- ROC curve --------------------------------------------------------
    figure; hold on; grid on;
    if ~isempty(fpTest)
        plot(fpTest, tpTest, 'LineWidth', 2);
        legText = sprintf('Test ROC (AUC = %.3f)', aucTest);
    else
        legText = 'Test ROC (AUC undefined)';
    end
    plot([0 1], [0 1], 'k--', 'LineWidth', 0.5);
    xlabel('False Positive Rate (FPR)');
    ylabel('True Positive Rate (TPR)');
    title('ROC Curve — External Test Set');
    legend({legText, 'Random'}, 'Location', 'SouthEast');
    exportgraphics(gcf, fullfile(savePath, 'ROC_TestSet.jpg'), 'Resolution', 300);

    % --- Export to Excel --------------------------------------------------
    testExcelFile = fullfile(savePath, char(json.outputFileNames.testExcelFile));

    TestResultsTable = table(testErrorRate, aucTest, ...
        'VariableNames', {'Error','AUC'}, ...
        'RowNames',      {'ExternalTest'});

    try
        writetable(TestResultsTable, testExcelFile, ...
            'Sheet', 'ExternalTest', 'WriteRowNames', true);
    catch ME
        error(['Failed to write test results Excel file "%s".\n' ...
               'Possible causes: file is open in Excel, insufficient disk space, or bad path.\n' ...
               'MATLAB error: %s'], testExcelFile, ME.message);
    end

    fprintf('External test results saved.\n');

end   % end of optional external test set block