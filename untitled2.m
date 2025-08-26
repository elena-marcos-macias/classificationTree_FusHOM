%==================================================================
% Script:    PCA_vs_tSNE_Visualization.m
% Purpose:   Load dataset, preprocess (handle missing values, encode
%            categorical features, standardize), then compute and
%            plot PCA and t-SNE side by side—with tuned t-SNE.
%==================================================================

%% --------- Setup paths and save directory ----------
addpath(genpath('./utils'));
addpath('./requirements');

savePath = './results/';
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

%% --------- Load and clean data ---------
T_Original = readtable(fullfile('./data','FUS_test6.xlsx'));
T_Original = rmmissing(T_Original);

%% --------- Select features and target ---------
target_columns = [1,22];
ignore_columns = [2,3];
T_Data = selectColumns(T_Original, target_columns, ignore_columns);
labels = categorical(T_Original.Death);

%% --------- Encode features → numeric matrix X ---------
varNames = T_Data.Properties.VariableNames;
isNum    = varfun(@isnumeric, T_Data, 'OutputFormat','uniform');

X_num    = table2array(T_Data(:, isNum));
catNames = varNames(~isNum);
catBlocks = cell(1, numel(catNames));
for k = 1:numel(catNames)
    cats        = categorical(string(T_Data.(catNames{k})));
    catBlocks{k}= dummyvar(cats);
end
X_cat = horzcat(catBlocks{:});

X = zscore([X_num, X_cat]);

%% --------- PCA and (tuned) t-SNE ---------
rng(0, 'twister');

% PCA
[~, score, ~, ~, explained] = pca(X);

% t-SNE with higher learning rate, stronger exaggeration, more PCA dims, and exact algorithm
%Y = tsne( ...
%    X, ...
%    'NumDimensions',    2, ...
%    'Perplexity',       30, ...
%    'LearnRate',        500, ...    % higher than default
%    'Exaggeration',     12, ...     % stronger early pull   
%    'NumPCAComponents', 15, ...    % keep more variance before t-SNE
%    'Algorithm',        'exact', ...% use the exact gradient (slower, sometimes better)
%    'Verbose',          1, ...
%    'Standardize',      false ...
%);

%BEST FOR THE TIME BEING
%Y = tsne(X,...
%    'Perplexity', 5);

Y = tsne(X,...
    'Perplexity', 5);


%% --------- Plot side-by-side ---------
h = figure('Name','PCA vs t-SNE','NumberTitle','off', ...
           'Position',[100 100 1000 450]);

subplot(1,2,1)
gscatter(score(:,1), score(:,2), labels, [], '.', 12)
xlabel(sprintf('PC1 (%.1f%%)', explained(1)))
ylabel(sprintf('PC2 (%.1f%%)', explained(2)))
title('PCA'), grid on, axis tight

subplot(1,2,2)
gscatter(Y(:,1), Y(:,2), labels, [], '.', 12)
xlabel('t-SNE 1'), ylabel('t-SNE 2')
title('t-SNE (tuned)'), grid on, axis tight

sgtitle('Dataset Projection: PCA vs. t-SNE')
