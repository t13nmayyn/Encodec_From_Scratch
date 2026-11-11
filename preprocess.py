
import torchaudio
import torch
import torchaudio.functional as F 
from torch import nn 
import torchaudio.transforms as T
import os 
import random

DATA_DIR=""
from torch.utils.data import Dataset,DataLoader
class AudioDataset(Dataset):
    def __init__(self,sample_rate:int,root_dir:str,audio_duration:int):
        self.sample_rate=sample_rate
        self.audio_duration=audio_duration
        self.file_paths=[]
        self.label_ids=[]
        folders=sorted([d  for d in os.listdir(root_dir)])
        for foldername in folders:
            class_dir=os.path.join(root_dir,foldername)
            for file in os.listdir(class_dir):
                if file.endswith(".mp3"):
                    self.file_paths.append(os.path.join(class_dir,file))
    def preprocess(self,stride:int):
        all_chunks=[]
        for audiofile in self.file_paths:
            waveform,sp_rate=torchaudio.load(audiofile)

            if waveform.shape[0]>1:
                waveform=waveform.mean(dim=0,keepdim=True)

            if self.sample_rate!=sp_rate:
                sampler=T.Resample(orig_freq=sp_rate,new_freq=self.sample_rate)
                waveform=sampler(waveform)

            total_samples=waveform.shape[1]
            window_samples=self.audio_duration*self.sample_rate
            moveforward=stride*self.sample_rate

            for j in range(0,total_samples,moveforward):
                remaining_sample=total_samples-j
                if remaining_sample<self.sample_rate:
                    break
                audio_chunk=waveform[:,j:j+window_samples]

                audio_len=audio_chunk.shape[1]
                if audio_len<window_samples:
                    pad_ammount=window_samples-audio_len
                    audio_chunk=nn.functional.pad(audio_chunk,(0,pad_ammount))

                all_chunks.append(audio_chunk)
            random.shuffle(all_chunks)
            self.audio_chunks=torch.stack(all_chunks)


    
    def __len__(self):
        return len(self.audio_chunks)

    def __getitem__(self,idx):
        chunk=self.audio_chunks[idx]
        return chunk















            

        


