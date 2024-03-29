# The document and code for ICDehazing

The code is for the paper, "Illumination Controllable Dehazing Network based on Unsupervised Retinex Embedding".

If you have any quesions, feel free to contact me. My <b> E-mail </b> and <b> WeChat </b> can be found at my homepage: [<A HREF="https://xiaofeng-life.github.io/">Homepage</A>]

Note: I will update this repo after the school server reboot. This is almost the final version of my code. So you can easily use it for your research.

-----------------------------------------------------------

## step 1: prepare dataset

Download datasets from websites or papers. Follow the organization form below.
```
├── dataset_name
    ├── train
        ├── hazy
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── clear
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── DCP (if you need)
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── val
        ├── hazy
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── clear
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── test
        ├── hazy
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── clear (if you have)
            ├── im1.jpg
            ├── im2.jpg
            └── ...
```

### With DCP Prior

The DCP pesudo-label can be generated offline.
This repo "https://github.com/He-Zhang/image_dehaze/" is easy to read.
It can be used for data generation. Or you can choose another repo. 
It is worthy to note that different repos may provide different implements.

### Without DCP Prior

Follow the organization form above.

-----------------------------------------------------------

## step 2: train with DCP

```
python train_ICDehazing.py --results_dir ../results/ICDehazing/4KDehazing_L2_DCP --img_w 256 --img_h 256 --train_batch_size 2 --dataset 4KDehazing --rec_loss L2 --prior_per True --prior_per_weight 1 --prior_decay 0.9 --model ICDehazing
```

```
python train_ICDehazing.py --results_dir ../results/ICDehazing/OTS_L2_DCP --img_w 256 --img_h 256 --train_batch_size 2 --dataset OTS --rec_loss L2 --prior_per True --prior_per_weight 1 --prior_decay 0.9 --model ICDehazing
```

**note: the val_dataloader (validation data) during training process can be chosen from trainset.
But it should not be chosen from the testset.** 

-----------------------------------------------------------

## step 3: train without DCP

When you use ICDehazing as a comparative experiment, you may not use the DCP prior on your own dataset.


```
python train_ICDehazing.py --results_dir ../results/ICDehazing/4KDehazing_L2_DCP --img_w 256 --img_h 256 --train_batch_size 2 --dataset 4KDehazing --rec_loss L2 --prior_per False --model ICDehazing
```

```
python train_ICDehazing.py --results_dir ../results/ICDehazing/OTS_L2_DCP --img_w 256 --img_h 256 --train_batch_size 2 --dataset OTS --rec_loss L2 --prior_per False --model ICDehazing
```

-----------------------------------------------------------

## step 4: test

```
python inference_ICDehazing.py --results_dir ../results/ICDehazing/4KDehazing_L2_DCP_val --img_w 256 --img_h 256 --pth_path ../results/ICDehazing/4KDehazing_L2_DCP/models/last_x2y.pth --dataset 4KDehazing --if_mul_alpha False
```
```
python inference_ICDehazing.py --results_dir ../results/ICDehazing/OTS_L2_DCP_val --img_w 256 --img_h 256 --pth_path ../results/ICDehazing/OTS_L2_DCP/models/last_x2y.pth --dataset 4KDehazing --if_mul_alpha False
```
-----------------------------------------------------------
## Some Visual dehazing results with different controllable parameters on SOTS outdoor (trained on OTS)

![](figures/1.png)

![](figures/2.png)

![](figures/3.png)

![](figures/4.png)