# CaseRec

This is a PyTorch implementation for paper: Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure

### Data

We take Tenrec as an example and provide the data processing code.
For the Tenrec dataset, the licence to acquire data is needed, and more details can be found in this [link](https://github.com/yuangh-x/2022-NIPS-Tenrec). 
After that, put the 'QB-video.csv' into 'data/Tenrec' and run following code to preprocess the Tenrec dataset:
```bash
python data/Tenrec/data_process_Tenrec.py
```

### Train recommender without data augmentation

```bash
python run_DT.py --data_name='Tenrec' --ckp=0 --batch_size=512 --epochs=400 --hidden_size=64 --lr=0.005 --dro_reg=0 --debias_evaluation_k=0 --use_exposure_data=0 --action_relabel
```

### Train user simulator

```bash
python run_RT.py --data_name='Tenrec' --ckp=0 --batch_size=512 --epochs=400  --hidden_size=64 --lr=0.005 --dro_reg=0 --debias_evaluation_k=0 --use_exposure_data=0
```

### Data Augmentation

- Strategy: *Random*
```bash
python run_augmentation.py --data_name='Tenrec' --batch_size=64 --aug_length=10  --hidden_size=64 --item_size=24655 --max_timestep=200 --model_path=<RT-model-path> --ratio 0.5
```
- Strategy: *Self-Improving*
```bash
python run_augmentation.py --data_name='Tenrec' --batch_size=64 --aug_length=10  --hidden_size=64 --item_size=24655 --max_timestep=200 --model_path=<RT-model-path> --ratio 0.5 --augment_strategy="bootstrap" --recommender_path=<DT-model-path>
```

### Run CaseRec

```bash
python run_DT.py --data_name='Tenrec' --ckp=0 --batch_size=512 --epochs=400 --hidden_size=64 --lr=0.005 --dro_reg=0 --debias_evaluation_k=0 --use_exposure_data=0 --use_exposure4training --augmentation_ratio 1 --action_relabel --augment_strategy=<Strategy>
```

### Reference
The code is implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec) and [DebiasedSR_DRO](https://github.com/nancheng58/DebiasedSR_DRO)