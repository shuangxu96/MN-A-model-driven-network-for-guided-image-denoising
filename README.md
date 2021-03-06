# [A model-driven network for guided image denoising](https://doi.org/10.1016/j.inffus.2022.03.006)

[Shuang Xu](https://shuangxu96.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), Jialin Wang, [Kai Sun](http://math.xjtu.edu.cn/info/1372/10164.htm)*, [Chunxia Zhang](http://gr.xjtu.edu.cn/web/cxzhang ), [Junmin Liu](http://gr.xjtu.edu.cn/web/junminliu ), [Junying Hu](https://math.nwu.edu.cn/info/1191/2772.htm)


Test
----------
1. Download all the files from this [hyperlink](https://drive.google.com/drive/folders/1-BS3MSU8J9kLCjswoRbjo3i5e7BK2cSB?usp=sharing). 
2. Double-click `dataset.zip` to extract data.
3. Download the network weights from [Pretrained weights](#pretrained-weights) and place `.pth` files in `./weight` folder.
4. Run [test_MN.py](test_MN.py) to reproduce the following tables. 

|<img src="figs/NIR_table.jpg" width="450px"/>|<img src="figs/Flash_table.jpg" width="450px"/>|
|:---:|:---:|
|<i>(a) RNS </i>|<i>(b) FAIP</i>|

The table shows the evaluation results on RNS and FAIP with Gaussian noise. The 1st, 2nd and 3rd best values are marked by bold, red and underline, respectively.

Test on your own images
----------
1. Place your images in `./eval/Flash/guidance` and `./eval/Flash/target`.
2. Download the network weights from [Pretrained weights](#pretrained-weights) and place `.pth` files in `./weight` folder.
3. Run [eval_MN.py](eval_MN.py) to reproduce the following images. 

|Guidance|Target|Denoised|
|:---:|:---:|:---:|
|<img src="figs/toys_guidance_crop.jpg" width="300px"/>|<img src="figs/toys_target_crop.jpg" width="300px"/>|<img src="figs/toys_crop.jpg" width="300px"/>|
|<img src="figs/sofa_guidance_crop.jpg" width="300px"/>|<img src="figs/sofa_target_crop.jpg" width="300px"/>|<img src="figs/sofa_crop.jpg" width="300px"/>|


Here are tested results on real-world .

Pretrained weights
----------

|Model|# layers|# filters| Modality|
|---|:--:|:---:|:---:|
|[MN](https://drive.google.com/file/d/1Z3TowUKxoAQr9g-vZz_F4f-WdZI7M21j/view?usp=sharing)     | 7 | 64 |RGB-NIR|
|[MN](https://drive.google.com/file/d/1T8OqTrHlAakKoDl1ZcZJhIS4d1RUfBay/view?usp=sharing)     | 7 | 64 |Nonflash-Flash|
|[MN-L](https://drive.google.com/file/d/1NMDN_w8d0F2WdmpuqKgYycecas8_L9iH/view?usp=sharing)| 3 | 32  |RGB-NIR|
|[MN-L](https://drive.google.com/file/d/18Tjcz5QYrkY32HFG-QDLQY2UGz3YbqEp/view?usp=sharing)| 3 | 32  |Nonflash-Flash|

Citation
----------
```BibTex
@article{xu2022_MN,
     author = {S. Xu, J. Zhang, J. Wang, K. Sun, C. Zhang, J. Liu, J. Hu},
     title = {A model-driven network for guided image denoising},
     journal = {Inf. Fus.},
     volume = {85},
     number = {},
     pages = {60--71},
     month = {Sep.},
     year = {2022},
}
```
