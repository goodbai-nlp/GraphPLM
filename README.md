# Requirements
+ python 3.8
+ pytorch 1.12
+ see requirements.yml for more

We recommend to use conda to manage virtual environments:
```
conda env create -f requirements.yml
```

# Data Processing

We follow [Spring](https://github.com/SapienzaNLP/spring) to preprocess AMR graphs:
```
# 1. install spring 
cd spring && pip install -e .
# 2. modify configuration file in configs/xx
# 3. processing data
bash run-preprocess.sh
```

# Training

```
bash train.sh
```