currentFolderPath = pwd;
erkekFolder = fullfile(currentFolderPath, 'sounds', 'male');  % Erkek ses dosyalarının bulunduğu klasör
kadinFolder = fullfile(currentFolderPath, 'sounds', 'female');  % Kadın ses dosyalarının bulunduğu klasör

% Her bir klasördeki dosyaların isimlerini alıyoruz
erkekFiles = dir(fullfile(erkekFolder, '*.m4a'));
kadinFiles = dir(fullfile(kadinFolder, '*.m4a'));

% Ses dosyasını spektrograma dönüştürmek için kullanılan parametreleri belirliyoruz
win = 128; % Pencere boyutu 128 olarak belirlenmiştir
hop = win/2; % Hop (adım) boyutu, pencere boyutunun yarısı (64) olarak belirlenmiştir
nfft = win; % FFT'nin (Fast Fourier Transform - Hızlı Fourier Dönüşümü) boyutu, pencere boyutuna eşit olarak belirlenmiştir


% Toplam dosya sayısını hesaplıyoruz
totalFiles = numel(erkekFiles) + numel(kadinFiles);

% XTrain ve YTrain dizileri için gerekli alanı ayırıyoruz
XTrain = zeros(64, 64, 1, totalFiles);
YTrain = zeros(totalFiles, 1);


% Erkek ses dosyalarını işleyin
for i = 1:numel(erkekFiles)

    %     sunum esnasinda sadece 10 adet sesi upload etmesi icin 
      if i == 10
         break;
      end

    % verilen pathdeki Ses dosyasını okuyoruz
    [ses, orneklem_orani] = audioread(fullfile(erkekFolder, erkekFiles(i).name));

    % Ses dosyasını spektrograma dönüştüyoruz
    [s, ~, ~] = spectrogram(ses, win, hop, nfft, orneklem_orani, 'yaxis');
    
    S = abs(s); % 's' değişkeninin mutlak değerinin alınması ve 'S' değişkenine atanması
    
    %    goruntuyu yeniden boyutlandiriyoruz
    img = imresize(S, [64 64]);

    % Görüntüyü ve etiketi XTrain ve YTrain dizilerine ekleyin
    XTrain(:, :, :, i) = img;

    %erkek sesinin label degeri 0 olacaktir
    YTrain(i) = 0;
    fprintf('%d / %d Erkek sesi içeri aktarılıyor lütfen bekleyiniz\n', i,numel(erkekFiles));
end


% Kadın ses dosyalarını işliyoruz
for i = 1:numel(kadinFiles)

%     sunum esnasinda sadece 10 adet sesi upload etmesi icin 
    if i == 10
        break;
    end
    % Ses dosyasını okuyoruz
    [ses, orneklem_orani] = audioread(fullfile(kadinFolder, kadinFiles(i).name));

    % Ses dosyasını spektrograma dönüştüyoruz
    [s, ~, ~] = spectrogram(ses, win, hop, nfft, orneklem_orani, 'yaxis');
    
    S = abs(s); % 's' değişkeninin mutlak değerinin alınması ve 'S' değişkenine atanması
    
    %    goruntuyu yeniden boyutlandiriyoruz
    img = imresize(S, [64 64]);

    % Görüntüyü ve etiketi XTrain ve YTrain dizilerine ekliyoruz
    XTrain(:, :, :, i + numel(erkekFiles)) = img;
     
    %     Kadın sesinin label degeri 0 olacaktir
    YTrain(i + numel(erkekFiles)) = 1;
    fprintf('%d / %d Kadın sesi içeri aktarılıyor lütfen bekleyiniz\n', i,numel(kadinFiles));
end



disp('Sesler başarıyla içeri aktarıldı');

% Veri ayırma oranı Burada %80i train %20'si test datasi olacaktir
trainRatio = 0.8;

% Karıştırma ve bölme için cvpartition nesnesi oluştur
c = cvpartition(totalFiles, 'HoldOut', 1 - trainRatio);


XTrain_final = XTrain(:, :, :, training(c)); % Eğitim seti olarak belirlenen görüntülerin seçilmesi
YTrain_final = categorical(YTrain(training(c))); % Eğitim seti olarak belirlenen etiketlerin seçilmesi ve kategorik formatına dönüştürülmesi

XTest = XTrain(:, :, :, test(c)); % Test seti olarak belirlenen görüntülerin seçilmesi
YTest = categorical(YTrain(test(c))); % Test seti olarak belirlenen etiketlerin seçilmesi ve kategorik formatına dönüştürülmesi


% Görüntülerin boyutu
inputSize = [64 64 1];

% Sınıf sayısı
numClasses = 2;

% CNN katmanlarını oluştur
layers = [ 
 imageInputLayer([64 64 1]) % Giriş katmanı, girdi olarak alınan görüntülerin boyutunu belirtir
    convolution2dLayer(3,8,'Padding','same') % 2D konvolüsyon katmanı, çekirdek boyutu 3 ve 8 filtre ile, sınırların aynı kalmasını sağlar
    batchNormalizationLayer % Küme normalleştirme katmanı, öğrenme sürecini hızlandırmak ve daha kararlı hale getirmek için 
    reluLayer % ReLU (Rectified Linear Unit) aktivasyon katmanı 

    maxPooling2dLayer(2,'Stride',2) % 2x2 boyutunda maksimum birleştirme (max pooling) katmanı, her 2 piksel adımla birlikte
    convolution2dLayer(3,16,'Padding','same') % 2D konvolüsyon katmanı, çekirdek boyutu 3 ve 16 filtre ile, sınırların aynı kalmasını sağlar
    batchNormalizationLayer % Küme normalleştirme katmanı
    reluLayer % ReLU aktivasyon katmanı

    maxPooling2dLayer(2,'Stride',2) % 2x2 boyutunda maksimum birleştirme (max pooling) katmanı, her 2 piksel adımla birlikte
    convolution2dLayer(3,32,'Padding','same') % 2D konvolüsyon katmanı, çekirdek boyutu 3 ve 32 filtre ile, sınırların aynı kalmasını sağlar
    batchNormalizationLayer % Küme normalleştirme katmanı
    reluLayer % ReLU aktivasyon katmanı

    fullyConnectedLayer(numClasses) % Tamamen bağlı (fully connected) katman, 'numClasses' sayısında nöron içerir
    softmaxLayer % Softmax katmanı, çıktıları olasılıklara dönüştürür
    classificationLayer % Sınıflandırma katmanı, çıktıları belirtilen sınıflara göre etiketler
];


% Eğitim seçeneklerini belirle
options = trainingOptions('sgdm', ... % Stokastik gradyan iniş momentum (sgdm) optimizasyon algoritması kullanılır
    'InitialLearnRate',0.01, ... % Başlangıç öğrenme oranı 0.01 olarak ayarlanmıştır
    'MaxEpochs',20, ... % Maksimum epoch (tur) sayısı 20 olarak belirlenmiştir
    'Shuffle','every-epoch', ... % Her epoch'ta verinin karıştırılması belirlenmiştir
    'Verbose',false, ... % Eğitim süresince ayrıntılı çıktıların görüntülenmemesi belirlenmiştir
    'Plots','training-progress'); % Eğitimin ilerleyişini gösteren bir çizim oluşturulması belirlenmiştir


% CNN'yi eğit
disp('Model eğitimi başlatıldı');
net = trainNetwork(XTrain_final, YTrain_final, layers, options);

% test verilerini predict ediyoruz
YPred = classify(net, XTest);

%  dogruluk degeri icin hesaplamalr yapiliyor
accuracy = sum(YPred == YTest)/numel(YTest);

fprintf('Doğruluk Oranı: %f\n', accuracy);

% Confusion Matriximi olusturuyoruz
confusionMatrix = confusionmat(YTest, YPred);

% Confusion Matrix'in Gösterimi
confusionchart(confusionMatrix);

%model kaydediliyor
path = fullfile(currentFolderPath, 'savingModel', 'lastTrainingModel.mat');  % Erkek ses dosyalarının bulunduğu klasör

save(path, 'net');
fprintf('%s modeliniz başarıyla kaydedilmiştir \n', path);

% Ses tanima programi calistiriliyor
sesTanimaProgramPath = fullfile(currentFolderPath, 'sestanima.m');  % Erkek ses dosyalarının bulunduğu klasör
run(sesTanimaProgramPath);