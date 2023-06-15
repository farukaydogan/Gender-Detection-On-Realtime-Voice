% Model yükleme
path = fullfile(currentFolderPath, 'savingModel', 'lastTrainingModel.mat');  % Erkek ses dosyalarının bulunduğu klasör

load(path, 'net');  %  model dosyanızın yolu olmalıdır

% Örnekleme oranını belirleme
orneklem_orani = 16000; % Hz

% Ses okuma nesnesi oluşturma
reader = audioDeviceReader('SampleRate', orneklem_orani, 'SamplesPerFrame', orneklem_orani);

% 3 saniyelik ses verisi için arabellek oluşturma
buffer = zeros(orneklem_orani * 3, 1);

% Ses dalga grafiği için figure oluşturma
fig1 = figure; % Yeni bir figür penceresi oluşturulur
subplot(2, 1, 1); % Oluşturulan figürü iki satır ve bir sütundan oluşan bir ızgara şeklinde alt grafiğe bölme ve ilk alt grafiği seçme
hPlot = plot(buffer); % 'buffer' verisini çizen bir çizim oluşturma ve çizim nesnesini 'hPlot' değişkenine atama

% gerekli textler hazirlaniliyor
title('Ses Dalga Grafiği'); % Çizime başlık ekleme: 'Ses Dalga Grafiği'
xlabel('Zaman (ms)'); % X ekseni için etiket ekleme: 'Zaman (ms)'
ylabel('Amplitüd'); % Y ekseni için etiket ekleme: 'Amplitüd'
ylim([-1 1]); % Y ekseninin limitlerini -1 ve 1 aralığına ayarlama

% Cinsiyet tahmini için figure oluşturuyoruz
subplot(2, 1, 2);
% font ayarlari yapiliyor
hText = text(0.5, 0.5, '', 'FontSize', 14);
% textler ayarlaniyor
title('Cinsiyet Tahmini');
% axis kapatiliyor
axis off;

% Sonsuz döngüyü başlatma
while ishandle(fig1)
    % 1 saniyelik ses verisini okuyoruz
    audioData = reader();

    % Ses verisini arabelleğe ekleniyor
    buffer = [buffer(orneklem_orani+1:end); audioData];

    % Ses dalga grafiğini güncelleniyor
    set(hPlot, 'YData', buffer);

    % Cinsiyet tahmini yapma ve sonucu gösterme
    gender = predictGender(buffer, orneklem_orani, net);
    set(hText, 'String', gender);

    % Çizimleri güncelleniyoruz
    drawnow;
end

% Ses okuma nesnesini temizleniyor
release(reader);


% predict fonksiyonu tanimlaniyor
function gender = predictGender(audioData, orneklem_orani, net)
    % Ses verisini spektrograma dönüştürme
    win = 128; % Pencere boyutu 128 olarak belirlenmiştir
    hop = win/2; % Hop (adım) boyutu, pencere boyutunun yarısı (64) olarak belirlenmiştir
    nfft = win; % FFT'nin (Fast Fourier Transform - Hızlı Fourier Dönüşümü) boyutu, pencere boyutuna eşit olarak belirlenmiştir

    %      sesi spektrograma ceviriyoruz
    [s, ~, ~] = spectrogram(audioData, win, hop, nfft, orneklem_orani, 'yaxis');

    %     karmasik sayilari ayirt etmek icin mutlak degerini aliyoruz
    S = abs(s);

    % fotografi tekrar boyutlandiriyoruz     
    img = imresize(S, [64 64]);

    % Model ile tahmin yapma
    label = classify(net, img);

    % Erkek veya Kadın olarak çıktıyı dönüştürme
    if label == categorical(0)
        gender = 'Erkek';
    elseif label == categorical(1)
        gender = 'Kadın';
    end
end