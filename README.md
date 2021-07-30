# ddackdae
this is easy-to-use ml package

I am planning to include simple models pre-trained on common dataset for using it in the future.

Here are model list I am planning to include in this project



### NLP

**model**

- Transformer (WIP)
- Bert



**dataset**

- translation dataset

  I don't know what happed in 2014, but dataset made by 2014 is really popular. if you want to know about dataset more, please check [paper with code machine translation Benchmarks](https://paperswithcode.com/task/machine-translation)

  | dataset name                       | train   | valid | test | How to get                                                   |
  | ---------------------------------- | ------- | ----- | ---- | ------------------------------------------------------------ |
  | WMT14 en-de (english to germany)   | 4508785 | 3000  | 3003 | [🤗Datasets](https://huggingface.co/datasets/wmt14)           |
  | IWSLT14 en-de (english to germany) | 160239  | 7283  | 6750 | [R-DROP github repo](https://github.com/dropreg/R-Drop/tree/main/fairseq_src/examples/translation_rdrop) |
  |                                    |         |       |      |                                                              |
  |                                    |         |       |      |                                                              |

  



### Computer Vision

**Image Classification**

- dataset
  - CIFAR 10, CIFAR 100, ImageNet

- Alexnet
- VGG
- ResNet
- Efficient Net
- ...?
- Vision Transformer(ViT)

**Face Recognition**

- dataset

  Train dataset : MS-Celeb-1M

  Test Datatset : MegaFace(MF1), IJB-C, LFW, YTF, CFP-FP

- arcFace 

- 💦Circle Loss : Computing Loss
  - https://github.com/TinyZeaMays/CircleLoss 
  - https://github.com/layumi/Person_reID_baseline_pytorch

