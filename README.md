# Requirements
+ python 3.8
+ pytorch 1.12
+ see requirements.yml for more

We recommend to use conda to manage virtual environments:
```
conda env create -f requirements.yml
```

# Pre-Training

```
cd pre-train
bash train-large.sh
```

# Fine-tuning (AMR2Text)
```
cd fine-tune/amr2text
bash run.sh
```

# Fine-tuning (AMRParsing)
```
cd fine-tune/text2amr
bash run.sh
```
