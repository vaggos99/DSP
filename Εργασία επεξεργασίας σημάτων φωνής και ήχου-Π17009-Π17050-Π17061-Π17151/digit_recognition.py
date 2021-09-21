import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import os
import scipy
import time
import sys       


def fVSb_accuracy(file_path,frame_length=2048, hop_length=1024):
    accuracy=0
    c=0
    for file in os.listdir(file_path):
        acc=0
        name=file.replace('.wav','')
        foreground_start_sample=int(name.split('_')[0])
        foreground_end_sample =int(name.split('_')[1])
        test_signal, sr = librosa.load(file_path+file,sr=44100)
        #calculate square energy
        se_test=find_se(test_signal,frame_length,hop_length)
        #calculate zero crossing rate per frame sample
        zcr_test=zero_crossing_rate(test_signal,frame_length,hop_length)
        #calculate the main frequency per sample
        freqs_test=get_freq(test_signal,frame_length,hop_length,sr)
        bVSf_test=bf_classifier(se_test,zcr_test,freqs_test)
        bVSf_test=scipy.signal.medfilt(bVSf_test, kernel_size=5)
        sample_bf=get_bf_classes_per_sample(bVSf_test,len(test_signal))
        for i in range(len(sample_bf)):
            if i<21729 or i>46138:
                if sample_bf[i]==0:
                    acc+=1
            else:
                if sample_bf[i]==1:
                    acc+=1
        acc=acc/len(test_signal)
        accuracy=accuracy+acc
        c=c+1
    print("Accuracy of background vs foreground:",accuracy/c)
                

def get_bf_classes_per_sample(bVSf,signal_size,frame_size=2048,hop_size=1024,sample_rate=44100):
    s_e=[]
    if bVSf[0]==1:
        s_e.append([0,frame_size]) 
    start=0
    end=frame_size
    for i in range(1,len(bVSf)):

        if bVSf[i-1]==1 and bVSf[i]==1:
            end=i*hop_size+frame_size
        elif bVSf[i-2]==1 and bVSf[i-1]==0 and bVSf[i]==1 and i-2>=0:
            end=i*hop_size+frame_size
        elif bVSf[i-1]==0 and bVSf[i]==1 and bVSf[i+1]==1:
            start=i*hop_size
            end=start+frame_size
        elif bVSf[i-1]==1 and bVSf[i]==0 and  bVSf[i+1]==1:
            end=i*hop_size+frame_size
        elif  bVSf[i-2]==1 and bVSf[i-1]==1 and bVSf[i]==0 and i-2>=0:
            end=i*hop_size+frame_size
            s_e.append([start,end])
    sample_bf=[0]*signal_size
    for k in s_e:
        if (k[1]-k[0])/sample_rate<0.100:
            continue
        sample_bf[k[0]:k[1]]=[1]*(k[1]-k[0])
    return sample_bf
    
  
def bandpass_filter(signal,low_freq,high_freq,sample_rate,order=2):
    nyq_freq=sample_rate/2
    low=low_freq/nyq_freq
    high=high_freq/nyq_freq
    b,a=scipy.signal.butter(order,[low,high],'bandpass',analog=False)
    signal=scipy.signal.filtfilt(b,a,signal,axis=0)
    return signal
    
def getData(file_path, frame_length, hop_length,mels=128,f_max=5000):

    train_dataset_X = []
    train_dataset_Y=[]

    for file in os.listdir(file_path):
        label = file.split('.')[0]
        wave, sample_rate = librosa.load(file_path+file,sr=44100)
        wave=bandpass_filter(wave,400,3400,sample_rate)
        mel_spec = librosa.feature.melspectrogram(wave, sample_rate, n_fft=frame_length, hop_length=hop_length ,n_mels=mels,fmax=f_max)
        mel_spec = librosa.amplitude_to_db(mel_spec)
        max_feature=mel_spec.max(axis=1, keepdims=False)
        train_dataset_X.append(max_feature)
        train_dataset_Y.append(label)

    return train_dataset_X,train_dataset_Y

def find_se(signal,frame_size,hop_size):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0,len(signal),hop_size): 
        rmse_current_frame =sum(signal[i:i+frame_size]**2)
        rmse.append(rmse_current_frame)
    return np.array(rmse)

def get_freq(signal,frame_size,hop_size,sample_rate):
    freqs=[]
    sp = np.fft.fft(signal)
    # find the main frequency for each frame
    for i in range(0,len(signal),hop_size): 
        frame=signal[i:i+frame_size]
        w = np.fft.fft(frame)
        f = np.fft.fftfreq(len(w),1/sample_rate)
        idx = np.argmax(np.abs(w))
        max_freq = f[idx]
        freqs.append(abs(max_freq))
    return freqs
        
    
def zero_crossing_rate(signal,frame_size,hop_size):
    zcr=[]
    #calculate zcr for each frame
    for i in range(0,len(signal),hop_size):
        frame=signal[i:i+frame_size]
        total=0
        for k in range(1,frame.size):
            s1=0
            s2=0
            if frame[k]>=0:
                s1=1
            elif frame[k]<0:
                s1=-1

            if frame[k-1]>=0:
                s2=1
            elif frame[k-1]<0:
                s2=-1
            total=total+abs(s1-s2)
        zcr_current_frame= total/(2*frame.size)
        zcr.append(zcr_current_frame)
    return np.array(zcr)

def bf_classifier(se,zcr,freqs):
    
    classes=[]
    se_thres=np.mean(se)/2 
    zcr_thres=np.mean(zcr)*3/2 
    for i in range(se.size):
        if zcr[i]<=zcr_thres and se[i]>=se_thres and freqs[i]<=3400:
            classes.append(1)
        else:
            classes.append(0)
       
            
    return classes
def get_the_foreground(bVSf,signal,frame_size=2048,hop_size=1024):
    numbers=[]
    start=0
    end=frame_size
    for i in range(1,len(bVSf)):

        if bVSf[i-1]==1 and bVSf[i]==1:
            end=i*hop_size+frame_size
        elif bVSf[i-2]==1 and bVSf[i-1]==0 and bVSf[i]==1 and i-2>=0:
            end=i*hop_size+frame_size
        elif bVSf[i-1]==0 and bVSf[i]==1 and bVSf[i+1]==1:
            start=i*hop_size
            end=start+frame_size
        elif bVSf[i-1]==1 and bVSf[i]==0 and  bVSf[i+1]==1:
            end=i*hop_size+frame_size
        elif  bVSf[i-2]==1 and bVSf[i-1]==1 and bVSf[i]==0 and i-2>=0:
            end=i*hop_size+frame_size
            numbers.append(signal[start:end])
       
    numbers = [x for x in numbers if not len(x)/sr<0.100]
    return numbers

def prediction_score(dataset_x,dataset_y):
    
    mean_success_rate=0
    kfold = KFold(3, shuffle=True)
    for train_index, test_index in kfold.split(dataset_x):
        x_train, x_test=dataset_x[train_index],dataset_x[test_index]
        y_train, y_test=dataset_y[train_index],dataset_y[test_index]
        knn = KNeighborsClassifier(n_neighbors = 3).fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        mean_success_rate+=accuracy_score(y_test, y_pred)
    print("Knn prediction score:",mean_success_rate/3)
        
if __name__ == "__main__":
    
    FRAME_SIZE = 2048
    HOP_SIZE=1024
    MELS=128
    FMAX=5000
    file_name=''
    num = int(input("Press 1, 2, 3 or 4 to choose an audio file.\n"))
    while num >4 or num<1:
        num = int(input("Press 1, 2, 3 or 4 to choose an audio file.\n"))
    if num==1:
        file_name='recording.wav'
    elif num==2:
        file_name='recording1.wav'
    elif num==3:
        file_name='recording2.wav'
    else:
        file_name='recording3.wav'
    
    #loading audio
    signal, sr = librosa.load('recordings/'+file_name,sr=44100)
    sd.play(signal, sr)
    time.sleep(1.5/sr*len(signal))
    
    #calculate square energy
    se=find_se(signal,FRAME_SIZE,HOP_SIZE)
    #calculate zero crossing rate per frame sample
    zcr=zero_crossing_rate(signal,FRAME_SIZE,HOP_SIZE)
    #calculate the main frequency per sample
    freqs=get_freq(signal,FRAME_SIZE,HOP_SIZE,sr)
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(signal)
    
    ax=plt.subplot(4, 1, 2)
    x_coordinate = [i for i in range(len(se))]
    ax.scatter(x_coordinate,se, s=10, c='b', marker="s", label='square energy')

    ax1 =plt.subplot(4, 1, 3)
    x_coordinate = [  i for i in range(len(zcr)) ]
    ax1.scatter(x_coordinate,zcr, s=10, c='b', marker="s", label='zero crossing rate')


    ax2 =plt.subplot(4, 1, 4)
    x_coordinate = [  i for i in range(len(freqs)) ]
    ax2.scatter(x_coordinate,freqs, s=10, c='b', marker="s", label='frequencies')
    plt.show()
    
    #classifying background vs foreground
    bVSf=bf_classifier(se,zcr,freqs)
    bVSf=scipy.signal.medfilt(bVSf, kernel_size=5)
    numbers=get_the_foreground(bVSf,signal)
    
    print("Number of digits:",len(numbers))
    if len(numbers)<5 :
        print("You gave less than 5 digits ")
        sys. exit()
    if  len(numbers)>10:
        print("You gave more than 10 digits ")
        sys. exit()
    #extracting the mel specs for each number and find the max value for each frequency   
    mel_spectrograms=[]
    for num in numbers:
        #apply a bandpass filter
        num=bandpass_filter(num,400,3400,sr)
        spec=librosa.feature.melspectrogram(num, sr=sr, n_fft=int(FRAME_SIZE), hop_length=int(HOP_SIZE), n_mels=MELS,fmax=FMAX)
        spec = librosa.amplitude_to_db(spec)
        max_spec=spec.max(axis=1, keepdims=False)
        mel_spectrograms.append(max_spec)
        
    #extracting the training dataset 
    train_dataset_x,train_dataset_y = getData('knn_training_data/',int(FRAME_SIZE),int(HOP_SIZE),mels=MELS)
    #scale data
    scaler=StandardScaler()
    scaled_data=np.concatenate((mel_spectrograms,train_dataset_x))
    scaled_data=scaler.fit_transform(scaled_data)
    mel_spectrograms=scaled_data[0:len(mel_spectrograms),:]
    train_dataset_x=scaled_data[len(mel_spectrograms):,:]
   
    
    #train and predict with KNN
    knn = KNeighborsClassifier(n_neighbors = 3).fit(train_dataset_x, train_dataset_y)
    knn_predictions = knn.predict(mel_spectrograms)
    print("The digits that you said:")
    print(*knn_predictions, sep = ", ")
    fVSb_accuracy('bVSf_test_data/')
    prediction_score(np.array(train_dataset_x),np.array(train_dataset_y))
