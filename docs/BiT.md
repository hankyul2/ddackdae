## Big Transfer: General Visual Representation Learning (BiT)

2020년 5월 쯤 아카이브에 등재된 논문인듯. 원래 읽을 생각이 없었는데 기존의 ViT 논문에서 비교 대상으로 지목하길래 읽고 정리볼께

[공식 레포](https://github.com/google-research/big_transfer/tree/master/bit_pytorch)

목표

1. 논문 요약
2. 구현
   1. 직접 짜보기
      1. Dataset
      2. Model
      3. Load Weight
      4. Train
   2. Fine-tuning 결과 재현하기
   3. Pre-training 결과 재현하기

### 요약

제안 방법 : pre-training and fine-tuning 과정을 잘 정리함

### 1. 또입

배경 : source dataset에서 학습하고 target dataset에서 fine-tuning 하는 방법은 적은 데이터셋에서도 높은 정확도를 달성하게 도와준다.

문제 : 기존까지 학습을 위한 많은 방법들이 제시되었다. 

제안 방법 : 우리는 이 학습 방법들을 잘 사용해서 Transfer learning을 해볼께! 우리가 사용하는 데이터셋의 크기에 따라서 각기 다른 크기의 모델을 만들 었어. 

| Dataset      | Dataset Size | Model Name | Architecture |
| ------------ | ------------ | ---------- | ------------ |
| JFT-300M     | 300M         | BiT-L      | ResNet 152x4 |
| ImageNet-21k | 14M          | BiT-M      |              |
| ILSVRC-2012  | 1.3M         | BiT-S      |              |

장점 : Pre-training을 한번 잘 해두면 여러개의 Task에 Transfer Learning을 적용할 수 있으니 이득이지. 그런데 Transfer Learning을 할 때 Hyper Parameter를 찾는데 시간이 오래 걸리잖아?? 내가 이부분에서 꿀팁을 줄께!



### 2. Big Transfer

Pre-training 시에 사용한 도구들(Upstream)과 Fine-tuning 시에 사용한 도구들(Downstream)을 소개할려고 해.

먼저 pre-training 시에 사용해야 하는 도구들이야. 

1. Scale: 큰 데이터셋에서는 큰 모델을. 작은 데이터셋에서는 작은 모델을 사용하자
2. Group Normalization (GN) Weight Standardization (WS) Batch Normalization (BN) : GN과 WS를 사용하는게 더 좋았어!

다음으로 Fine-tuning 시에 사용하는 도구들이얌.

1. 하이퍼파라미터 정하기: BiT-HyperRule이라는 가장 중요한 하이퍼파라미터를 정하는 방법을 사용했어. 하이퍼 퍼라미터를 정하는데 중요한 요소는 이미지의 크기(resolution)과 데이터셋의 크기야. 이 요소를 사용해서 정하는건 트레인 스케쥴 시간, 이미지 크기, MixUp regularization 이야. 
2. 데이터 전처리 : Train : Resize, Crop, Horizontal Flip, Test : Resize
3. Regularization Methods : no weight decay, no init param, no dropout



### 3. 실험(3.3절까지)

모델의 경우 ResNet V2를 사용했어. 기존 모델과 다른 점이 있다면 Group Normalization과 Weight Standardization이지. 그리고 hidden layer의 개수를 4배 가량 늘렸어.

**Pre-training**

| 범주      | 세부 카테고리           | 내용                                                         | 중요한 부분                                                  |
| --------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 데이터    | 데이터전처리            | resize, crop(224, 224), normalization                        |                                                              |
|           | Train Data Augmentation | scale: Nope, geometric: horizontal flip, random crop, color: Nope |                                                              |
|           | Test Data Augmentation  | scale: Nope, geometric: Nope, color: Nope                    |                                                              |
| 모델      | 모델 구조               | ResNet154 version 2 with widening factor 4                   | 데이터셋의 크기가 커짐에 따라서 모델의 크기 또한 커져야 한다. |
|           | Regularization          | group normalization, weight standardization                  | 기존의 BN 대신 사용                                          |
| 학습 도구 | optimizer               | SGD with momentum, weight decay                              |                                                              |
|           | criterion               | cross entropy loss                                           |                                                              |
|           | learning rate scheduler | init lr = 0.03, decay 10 at [30, 60, 80], warmup for 5000 steps |                                                              |
| 학습      | Batch size              | 4096                                                         |                                                              |
|           | epoch                   | 90                                                           |                                                              |
|           | GPU Device              | TPUv3 512대 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ |                                                              |
| 평가      | 평가 방법               |                                                              |                                                              |



**Fine-tuning**

| 범주      | 세부 카테고리           | 내용                                                         | 중요한 부분                                                  |
| --------- | ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 데이터    | 데이터전처리            | resize, crop, normalization                                  | (160, 160) -> (128, 128)<br />(448, 448) -> (384, 384)       |
|           | Train Data Augmentation | scale: Nope, geometric: horizontal flip, random crop, color: Mixup(데이터마다 다름) | 데이터셋의 크기에 다라서 달라진다.                           |
|           | Test Data Augmentation  | scale: Nope, geometric: Nope, color: Nope                    |                                                              |
| 모델      | 모델 구조               | ResNet154 version 2 with widening factor 4                   | 데이터셋의 크기가 커짐에 따라서 모델의 크기 또한 커져야 한다. |
|           | Regularization          | group normalization, weight standardization                  | 기존의 BN 대신 사용                                          |
| 학습 도구 | optimizer               | SGD with momentum(0.9), weight decay                         |                                                              |
|           | criterion               | cross entropy loss                                           |                                                              |
|           | learning rate scheduler | init lr = 0.003, decay 10 at 30%, 60% 90%                    |                                                              |
| 학습      | Batch size              | 512                                                          |                                                              |
|           | epoch                   | less than 20k : 500 step<br />less than 500k : 10k step<br >more than 500k : 20k step | 데이터셋의 크기에 따라서 달라진다.                           |
|           | GPU Device              | TPUv3 512대 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ |                                                              |
| 평가      | 평가 방법               |                                                              |                                                              |

