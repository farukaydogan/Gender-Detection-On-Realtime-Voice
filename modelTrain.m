erkekFolder = '/Users/pc/Documents/MATLAB/sounds/male';  % Erkek ses dosyalarının bulunduğu klasör
kadinFolder = '/Users/pc/Documents/MATLAB/sounds/female';  % Kadın ses dosyalarının bulunduğu klasör

% Her bir klasördeki dosyaların isimlerini alın
erkekFiles = dir(fullfile(erkekFolder, '*.m4a'));
kadinFiles = dir(fullfile(kadinFolder, '*.m4a'));

% Ses dosyasını spektrograma dönüştür
win = 128; % pencere boyutu
hop = win/2; % hop boyutu
nfft = win;

% Toplam dosya sayısını belirleyin
totalFiles = numel(erkekFiles) + numel(kadinFiles);

% XTrain ve YTrain dizileri için yer ayırın
XTrain = zeros(64, 64, 1, totalFiles);
YTrain = zeros(totalFiles, 1);

% Erkek ses dosyalarını işleyin
for i = 1:numel(erkekFiles)

%      if i == 10
%         break;
%     end
    % Ses dosyasını okuyun
    [ses, orneklem_orani] = audioread(fullfile(erkekFolder, erkekFiles(i).name));

    % Ses dosyasını spektrograma dönüştürün
    [s, ~, ~] = spectrogram(ses, win, hop, nfft, orneklem_orani, 'yaxis');
    S = abs(s);
    img = imresize(S, [64 64]);

    % Görüntüyü ve etiketi XTrain ve YTrain dizilerine ekleyin
    XTrain(:, :, :, i) = img;
    YTrain(i) = 0;
    fprintf('%d / %d Erkek sesi içeri aktarılıyor lütfen bekleyiniz\n', i,numel(erkekFiles));
end


% Kadın ses dosyalarını işleyin
for i = 1:numel(kadinFiles)

%      if i == 10
%         break;
%     end
    % Ses dosyasını okuyun
    [ses, orneklem_orani] = audioread(fullfile(kadinFolder, kadinFiles(i).name));

    % Ses dosyasını spektrograma dönüştürün
    [s, ~, ~] = spectrogram(ses, win, hop, nfft, orneklem_orani, 'yaxis');
    S = abs(s);
    img = imresize(S, [64 64]);

    % Görüntüyü ve etiketi XTrain ve YTrain dizilerine ekleyin
    XTrain(:, :, :, i + numel(erkekFiles)) = img;
    YTrain(i + numel(erkekFiles)) = 1;
    fprintf('%d / %d Kadın sesi içeri aktarılıyor lütfen bekleyiniz\n', i,numel(kadinFiles));
end



disp('Sesler başarıyla içeri aktarıldı');

% Veri ayırma oranı
trainRatio = 0.8;

% Karıştırma ve bölme için cvpartition nesnesi oluştur
c = cvpartition(totalFiles, 'HoldOut', 1 - trainRatio);

XTrain_final = XTrain(:, :, :, training(c));
YTrain_final = categorical(YTrain(training(c)));
XTest = XTrain(:, :, :, test(c));
YTest = categorical(YTrain(test(c)));


% Görüntülerin boyutu
inputSize = [64 64 1];

% Sınıf sayısı
numClasses = 2;

% CNN katmanlarını oluştur
layers = [ 
    imageInputLayer([64 64 1]) % Giriş katmanı, görüntülerin boyutunu belirtir
    convolution2dLayer(3,8,'Padding','same') 
    batchNormalizationLayer 
    reluLayer 

    maxPooling2dLayer(2,'Stride',2) 
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];


% Eğitim seçeneklerini belirle
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');


% CNN'yi eğit
disp('Model eğitimi başlatıldı');
net = trainNetwork(XTrain_final, YTrain_final, layers, options);

YPred = classify(net, XTest);

accuracy = sum(YPred == YTest)/numel(YTest);

fprintf('Doğruluk Oranı: %f\n', accuracy);

% Confusion Matrix
confusionMatrix = confusionmat(YTest, YPred);

% Confusion Matrix'in Gösterimi
confusionchart(confusionMatrix);

%model kaydediliyor
path="/Users/pc/Documents/MATLAB/savingModel/lastTrainingModel.mat";
save(path, 'net');
fprintf('%s modeliniz başarıyla kaydedilmiştir \n', path);