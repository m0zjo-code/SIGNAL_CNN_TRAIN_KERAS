#!/bin/bash

for i in `seq 1 10`;
do
python3 CNN_1D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/MeanPSDTrainingData.npz --prefix MeanPSD

python3 CNN_1D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/MaxPSDTrainingData.npz --prefix MaxPSD

python3 CNN_1D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/MinPSDTrainingData.npz --prefix MinPSD

python3 CNN_1D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/VarTrainingData.npz --prefix VarPSD
done 


python3 CNN_2D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/MagSpecTrainingData.npz --prefix MagSpec

python3 CNN_2D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/CecTrainingData.npz --prefix CecSpec

python3 CNN_2D_Single.py --input /home/jonathan/signaldoctor-zmqserver/nnetsetup/FFTTrainingData.npz --prefix FFTSpec
