#!/bin/bash

echo "Augmenting data..."
python $TAMPERING_HOME/tampering/rm/rrm/dataset/1_augmentation.py

echo "Labeling rewards..."
python $TAMPERING_HOME/tampering/rm/rrm/dataset/2_reward_calculation.py

echo "Filtering data..."
python $TAMPERING_HOME/tampering/rm/rrm/dataset/3_filtering.py
rm $TAMPERING_HOME/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_ver5_explicit_augmentation.json

echo "Preprocessing data..."
python $TAMPERING_HOME/tampering/rm/rrm/dataset/4_data_preprocess.py
rm $TAMPERING_HOME/datasets/hhrlhf/rm/train/hhrlhf_RM_5120_pref_ver5_explicit_augmentation_rrm.json
