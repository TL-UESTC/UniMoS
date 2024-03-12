# UniMoS
Pytorch implementation of **[Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation](https://arxiv.org/abs/2403.06946)** *(CVPR'24)*

![image](https://github.com/TL-UESTC/UniMoS/assets/68037940/057f1df9-2b2f-4476-a104-1c2361bc7a45)

- Requirements
```
python==3.8
pytorch==1.12
```

- How to run:
1. Install [CLIP](https://github.com/openai/CLIP)
2. Put Office-Home dataset under `./dataset/OfficeHome`
3. run by `python main.py --source [SOURCE] --target [TARGET]`
4. Check results in `./log/`
