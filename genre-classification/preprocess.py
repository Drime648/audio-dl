import os
import librosa
import math
import json
DATASET_PATH = "./Data/genres_original"
JSON_PATH = "./Data/data.json"
SAMPLE_RATE = 22050 # num samples per seconds
DURATION = 30 # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(datset_path, json_path, n_mfccs=13,n_fft=2048,hop_length=512,n_segments=5):
    

    #make dict to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment = int(SAMPLES_PER_TRACK / n_segments)
    expected_n_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    for i, (dirpath, dirnames, fnames) in enumerate(os.walk(datset_path)):
        if i > 0:

            #print (i, dirpath)
            components = dirpath.split("\\")
            print("\n processing the " + components[1])
            data["mapping"].append(components[1])

            for k in fnames:
                f_path = os.path.join(dirpath, k)
                signal, sr = librosa.load(f_path, sr=SAMPLE_RATE)
                for s in range(n_segments):
                    start_sample= s * num_samples_per_segment
                    finish_sample = start_sample + num_samples_per_segment

                    
                    

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                    sr, n_mfcc=n_mfccs, n_fft=n_fft,hop_length=hop_length)

                    mfcc = mfcc.T

                    if len(mfcc) == expected_n_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment: {}".format(f_path,s+1))
    

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent = 4)




            


save_mfcc(DATASET_PATH, JSON_PATH, n_segments=10)


    

