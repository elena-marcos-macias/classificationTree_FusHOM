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
%    decisionTree.*          – Full-model tree graph (vector PDF or SVG)
%    <excelFileName>.xlsx
%      · Errors              – Mean, SD, best, worst, median CV metrics
%      · PredictorImportance – Gini importance from the full model
%      · CVTreeImportance    – Importance from every CV training tree
%      · CVTreeSummary       – Aggregate stats across all CV trees
%      · ExternalTest        – Error rate + AUC on the held-out test file
%    Histograms.jpg          – Error-rate and AUC distributions across runs
%    ROCcurves.jpg           – Best / median / mean / worst ROC curves
%    Confusion_TestSet.jpg   – Confusion matrix for the external test set
%    ROC_TestSet.jpg         – ROC curve for the external test set
%
%  DEPENDENCIES
%  ------------
%    Statistics and Machine Learning Toolbox (fitctree, perfcurve, etc.)
%    selectColumns()  – custom helper in ./utils/
%
%  Author : [Your Name]
%  Date   : [Date]
% =========================================================================


%% ---- 1. PATH & OUTPUT FOLDER SETUP -------------------------------------
addpath(genpath('./data'));
addpath(genpath('./utils'));
addpath('./requirements');

savePath = './results/';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end


%% ---- 2. LOAD CONFIGURATION ---------------------------------------------
json = readstruct("data/instructionsDecisionTree.json");


%% ---- 3. LOAD & PREPARE TRAINING DATA -----------------------------------
% Read the training Excel file; drop any row that contains a missing value
% (listwise deletion keeps the design matrix rectangular and consistent).
trainingDataFile = char(json.inputDataSelection.trainingDataFile);
T_Original = rmmissing(readtable(['./data/' trainingDataFile]));

% Keep only the columns specified by the JSON include/exclude criteria
T_Data = selectColumns(T_Original, ...
    json.inputDataSelection.columnCriteria.target_columns, ...
    json.inputDataSelection.columnCriteria.ignore_columns);

catVariable       = char(json.inputDataSelection.catVariable);
T_ResultsVariable = T_Original.(json.inputDataSelection.resultsVariable);
nObs              = height(T_Data);

% --- Positive class validation -------------------------------------------
% The positive class is declared explicitly in the JSON instead of being
% inferred automatically (e.g. as the minority class), so that ROC
% orientation is deterministic across runs and data subsets.
positiveClass = string(json.inputDataSelection.positiveClass);
classNames    = unique(string(T_ResultsVariable));

if ~any(classNames == positiveClass)
    error(['Positive class "%s" not found in the response variable. ' ...
           'Available classes: %s'], positiveClass, strjoin(classNames, ', '));
end

negativeClasses = classNames(classNames ~= positiveClass);
fprintf('Positive class : %s\n',         positiveClass);
fprintf('Negative class : %s\n',         strjoin(negativeClasses, ', '));
fprintf('Total samples  : %d\n\n',       nObs);

% Binary ground-truth vector used for perfcurve (true = positive class)
trueLabels = string(T_ResultsVariable);
trueBinary = (trueLabels == positiveClass);


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
exportgraphics(treeFigRaster, fullfile(savePath, 'DecisionTree.jpg'), ...
    'Resolution', 300);

fprintf('Tree exported: DecisionTree.jpg\n');
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
    CVMdl = fitctree(T_Data, T_ResultsVariable, ...
        'KFold',                nFolds, ...
        'CategoricalPredictors',{catVariable}, ...
        'MinParentSize',        3);

    % --- Run-level performance (aggregated over all folds) ----------------
    [~, Score]    = kfoldPredict(CVMdl);  % out-of-fold predicted scores
    missClassRate = kfoldLoss(CVMdl);     % out-of-fold error rate

    % Locate the positive class column in the score matrix.
    % Computed once per run because CVMdl.ClassNames is consistent within a
    % run (MATLAB preserves the training-data class order).
    cvClassNames = string(CVMdl.ClassNames);
    posClassIdx  = find(cvClassNames == positiveClass, 1);
    if isempty(posClassIdx)
        error(['Positive class "%s" not found in CVMdl.ClassNames (run %d). ' ...
               'Verify the JSON label matches the data exactly.'], ...
               positiveClass, run);
    end

    [fp, tp, ~, auc] = perfcurve(trueBinary, Score(:, posClassIdx), true);

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

        % Derive actual (not approximate) train / test counts from the
        % partition object so per-tree metadata is exact.
        testIdx      = test(partition, fold);
        nTestActual  = sum(testIdx);
        nTrainActual = nObs - nTestActual;

        CVTreeImportanceRaw(treeCounter, :) = predictorImportance(cvTree);
        CVTreeRunID(treeCounter)            = run;
        CVTreeFoldID(treeCounter)           = fold;
        CVTreeTrainSize(treeCounter)        = nTrainActual;
        CVTreeTestSize(treeCounter)         = nTestActual;
    end
end


%% ---- 7. CV-TREE IMPORTANCE TABLES ---------------------------------------

% --- Full detail table: one row per CV training tree ---------------------
CVTreeImportanceTable = array2table( ...
    [CVTreeRunID, CVTreeFoldID, CVTreeTrainSize, CVTreeTestSize, CVTreeImportanceRaw], ...
    'VariableNames', [{'Run','Fold','TrainSize','TestSize'}, allPredictors]);

% --- Summary statistics across all nRuns × nFolds training trees ---------
meanImpCV = mean(CVTreeImportanceRaw, 1);
stdImpCV  = std(CVTreeImportanceRaw,  0, 1);
medImpCV   = median(CVTreeImportanceRaw, 1);
usageRate  = mean(CVTreeImportanceRaw > 0, 1);

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

writetable(ErrorTable,               excelFileName, 'Sheet','Errors',              'WriteRowNames', true);
writetable(PredictorImportanceTable, excelFileName, 'Sheet','PredictorImportance');
writetable(CVTreeImportanceTable,    excelFileName, 'Sheet','CVTreeImportance');
writetable(CVTreeSummaryTable,       excelFileName, 'Sheet','CVTreeSummary',      'WriteRowNames', true);

% Write the cell row immediately after the table:
% Row 1 = header, rows 2-5 = Mean / SD / Median / UsageRate, so A6 is the next row.
writecell(CVTreeSummaryValuesRange, excelFileName, ...
    'Sheet', 'CVTreeSummary', 'Range', 'A6');

fprintf('Results exported to: %s\n', excelFileName);


%% ---- 11. HISTOGRAMS (error rate & AUC distributions) -------------------
figure;

% Helper to add padding and aligned tick marks to a histogram axes
function styleHistogram(h, ax)
    binW = h.BinWidth;
    ylim(ax, [0, max(h.Values) * 1.15]);                           % 15 % headroom
    xlim(ax, [min(h.BinEdges) - 0.5*binW, max(h.BinEdges) + 0.5*binW]);
    xticks(ax, h.BinEdges);
end

ax1 = subplot(1, 2, 1);
h1  = histogram(ax1, errorResults(:,1), 'NumBins', 10);
title(ax1, 'ErrorRate Distribution');
%xlabel(ax1, 'Error Rate'); ylabel(ax1, 'Count');
grid(ax1, 'on'); pbaspect(ax1, [1 1 1]);
styleHistogram(h1, ax1);

ax2 = subplot(1, 2, 2);
h2  = histogram(ax2, errorResults(:,2), 'NumBins', 10);
title(ax2, 'AUC Distribution');
%xlabel(ax2, 'AUC'); ylabel(ax2, 'Count');
grid(ax2, 'on'); pbaspect(ax2, [1 1 1]);
styleHistogram(h2, ax2);

exportgraphics(gcf, fullfile(savePath, 'Histograms.jpg'), 'Resolution', 300);


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

exportgraphics(gcf, fullfile(savePath, 'ROCcurves.jpg'), 'Resolution', 300);
fprintf('\nAll figures and tables saved to: %s\n', savePath);


%% ---- 13. EXTERNAL TEST SET EVALUATION -----------------------------------
testDataFile   = char(json.inputDataSelection.testDataFile);
T_TestOriginal = rmmissing(readtable(['./data/' testDataFile]));

if isempty(T_TestOriginal)
    error('Test dataset is empty after loading/rmmissing(). Check the file: %s', testDataFile);
end

% Apply the same column-selection logic used on the training set
T_TestData          = selectColumns(T_TestOriginal, ...
    json.inputDataSelection.columnCriteria.target_columns, ...
    json.inputDataSelection.columnCriteria.ignore_columns);
T_TestResultsVariable = T_TestOriginal.(json.inputDataSelection.resultsVariable);

if isempty(T_TestResultsVariable)
    error('Test labels are empty. Check resultsVariable in JSON or the test file.');
end

testLabelsTrue = string(T_TestResultsVariable);
testBinary     = (testLabelsTrue == positiveClass);  % binary ground truth

fprintf('\n===== TEST SET CLASS DISTRIBUTION =====\n');
disp(tabulate(testLabelsTrue));

% --- Predict with the full model -----------------------------------------
[testPredLabels, testScores] = predict(Mdl, T_TestData);
testPredLabels = string(testPredLabels);

% --- Error rate -----------------------------------------------------------
testErrorRate = mean(testPredLabels ~= testLabelsTrue);

% --- AUC ------------------------------------------------------------------
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

% --- Report results -------------------------------------------------------
fprintf('\n===== EXTERNAL TEST SET RESULTS =====\n');
fprintf('Test samples : %d\n',   numel(testLabelsTrue));
fprintf('Error rate   : %.4f\n', testErrorRate);
if isnan(aucTest)
    fprintf('AUC          : undefined (single class in test set)\n\n');
else
    fprintf('AUC          : %.4f\n\n', aucTest);
end

% --- Confusion matrix -----------------------------------------------------
figure;
confusionchart(testLabelsTrue, testPredLabels);
title('External Test Set — Confusion Matrix');
exportgraphics(gcf, fullfile(savePath, 'Confusion_TestSet.jpg'), 'Resolution', 300);

% --- ROC curve ------------------------------------------------------------
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

% --- Export to Excel (new file) -----------------------------------------
testExcelFile = fullfile(savePath, char(json.outputFileNames.testExcelFile)); % call the new excel file

TestResultsTable = table(testErrorRate, aucTest, ...
    'VariableNames', {'Error','AUC'}, ...
    'RowNames',      {'ExternalTest'});
writetable(TestResultsTable, testExcelFile, ...
    'Sheet', 'ExternalTest', 'WriteRowNames', true);

fprintf('External test results saved.\n');