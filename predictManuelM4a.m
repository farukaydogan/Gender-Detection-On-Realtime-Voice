load('/Users/pc/Documents/MATLAB/savingModel/lastTrainingModel.mat', 'net');  % 'model_path.mat' model dosyanızın yolu olmalıdır

%kadin kendi mikrofon predict
[audioData, orneklem_orani] = audioread('/Users/pc/Documents/MATLAB/kadin.wav');  % 'audio_path.wav' ses dosyanızın yolu olmalıdır

%erkek kendi mikrofon predict
%[audioData, orneklem_orani] = audioread('/Users/pc/Documents/MATLAB/1.wav');  % 'audio_path.wav' ses dosyanızın yolu olmalıdır

%train data erkek predict
%[audioData, orneklem_orani] = audioread('/Users/pc/Documents/MATLAB/sounds/male/3.m4a');  % 'audio_path.wav' ses dosyanızın yolu olmalıdır


%train data kadin predict
%[audioData, orneklem_orani] = audioread('/Users/pc/Documents/MATLAB/sounds/female/1.m4a');  % 'audio_path.wav' ses dosyanızın yolu olmalıdır


% Ses dosyasını spektrograma dönüştür
win = 128; % pencere boyutu
hop = win/2; % hop boyutu
nfft = win;

% Spectrogram oluşturma
[s, ~, ~] = spectrogram(audioData, win, hop, nfft, orneklem_orani, 'yaxis');

S = abs(s);
img = imresize(S, [64 64]);

% 64x64 boyutuna yeniden boyutlandırma


% Model için ses verisini uygun şekle dönüştürme
%audioDataReshaped = reshape(s, [64, 64, 1]);  % modelin beklediği boyuta dikkat edin

% Model ile tahmin yapma


YPred = classify(net, img);

if YPred == categorical(0)
    disp('Erkek');
elseif YPred == categorical(1)
    disp('Kadın');
else
    disp('Bilinmeyen');
end

% Erkek veya Kadın olarak çıktıyı dönüştürme
% if label == 0
%     disp('Erkek');
% elseif label == 1
%     disp('Kadın');
% else
%     disp('Bilinmeyen');
% end