#!/bin/bash
for folder in ArticularyWordRecognition AtrialFibrillation CharacterTrajectories Cricket FaceDetection FingerMovements MotorImagery SelfRegulationSCP1 SelfRegulationSCP2 UWaveGestureLibrary
do
  for i in {1..5}
  do
    save_path=exp/$folder/$i
    python main.py \
    --dataset UCR \
    --UCR_folder $folder \
    --data_path data/UCR_UEA/Multivariate_arff/ \
    --device cuda:0 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --save_path $save_path \
  done
done
