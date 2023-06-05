% Model yükleme
load('/Users/pc/Documents/MATLAB/savingModel/lastTrainingModel.mat', 'net');  % 'model_path.mat' model dosyanızın yolu olmalıdır

% Örnekleme oranını belirleme
orneklem_orani = 16000; % Hz

% Mikrofon için ses kayıt nesnesi oluşturma
recorder = audiorecorder(orneklem_orani, 16, 1);

disp('Ses Kaydi Basladi')

% Kayıt başlangıcı
recordblocking(recorder, 5);  % 2 saniye kayıt


% Ses verisini alma
audioData = getaudiodata(recorder);

% Ses dosyasını spektrograma dönüştür
win = 128; % pencere boyutu
hop = win/2; % hop boyutu
nfft = win;

% Spectrogram oluşturma
[s, ~, ~] = spectrogram(audioData, win, hop, nfft, orneklem_orani, 'yaxis');

S = abs(s);
img = imresize(S, [64 64]);

% Model ile tahmin yapma
label = classify(net, img);


% Erkek veya Kadın olarak çıktıyı dönüştürme
if label == categorical(0)
    disp('Erkek');
elseif label == categorical(1)
    disp('Kadın');
else
    disp('Bilinmeyen');
end
% Spektrogramı çizme
spectrogram(audioData, win, hop, nfft, orneklem_orani, 'yaxis');
title(label);
drawnow;
% % Sonraki 2 saniyelik kayıt için hazırlık
% recordblocking(recorder, 2);