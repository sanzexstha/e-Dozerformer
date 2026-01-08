# e- Dozerformer


## Train and Test
1. Install the required packages: `pip install -r requirements.txt`
2. Data are publicly available at [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).
3. To reproduce the experimental results presented in the paper. Simply run the scripts at "/scripts/" as follows:
   ```
   bash ./scripts/ETTh1.sh
   bash ./scripts/ETTh2.sh
   bash ./scripts/ETTm1.sh
   bash ./scripts/ETTm2.sh
   bash ./scripts/electricity.sh
   bash ./scripts/Exchange_Rate.sh
   bash ./scripts/Traffic.sh
   bash ./scripts/WTH.sh
   bash ./scripts/ILI.sh
   ```
   


## Acknowledgements
We sincerely appreciate the foundational code from the following GitHub repositories: \
https://github.com/wanghq21/MICN \
https://github.com/zhouhaoyi/Informer2020 \
https://github.com/Thinklab-SJTU/Crossformer \
https://github.com/thuml/Time-Series-Library \
https://github.com/cure-lab/SCINet
