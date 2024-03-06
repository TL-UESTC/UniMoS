# UniMoS
Pytorch implementation of **Split to Merge: Unifying Separated Modalities for Unsupervised Domain Adaptation** *(CVPR'24)*

![image](https://github.com/TL-UESTC/UniMoS/assets/68037940/9ccf16a0-8442-472c-851f-6b50093dc64c)

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
