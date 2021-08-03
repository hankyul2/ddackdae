---
typora-copy-images-to: pics
---

## Domain Adaptation 

Source Domain : 일반 얼굴 이미지 (MS1-MV3)

Target Domain : 마스크 쓴 얼굴 이미지 



### BSP (2019)

2번 X, 3.1만 함 3.2는 이를 기존 모델에 적용하는 방법이다. 따라서 기존 모델에 대한 이해가 필요하다.

1. BSP의 주요 용어

   - DANN: Domain Adversarial Neural Network 
   - CDAN: Conditional Domain Adversarial Network

2. 요약

   - Adversarial Domain Adaptation : Learning Transferable Representation for knowledge transfer across domain
   - Adversarial Learning : strengthen transferability
   - Problem : discriminability has not fully explored before.
   - approach
     - spectral analysis of the feature representations -> 'deterioration of the discriminability'
     - *Our key finding is that the eigenvectors with the largest singular values will dominate the feature transferbility. As a consequence, the transferability is enhanced at the expense of over penalization of ther eigenvectors that embody rich structures crucial for discriminability*
     - *Towards this problem, we present Batch Spectral Penalization (BSP), a general approach to penalizing the largest singular values so that other eigenvectors can be relatively strengthed to boost the feature discriminability*
   - experiment : *yield state of the art results*

3. 도입

   1. *a domain discriminator is trained to distinguish the source from the target while feature representations are learned to confuse it simultaneously*
   2. singular value decomposition (SVD) to analyze the spectral properties of feature representations in batch

4. Transferability vs Discriminability (reason on discriminability loss)

   1. unsupervised domain adaptation problem : labelled source domain, unlabeled target domain

      Adversarial domain adaptation 중에서 유명한 논문

      1. Ganin at el. (2016) [Domain-adversarial training of neural network paper](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Domain-adversarial+training+of+neural+network&btnG=)
      2. Ganin at el. (2015) [Unsupervised domain adaptation by backpropagation paper](http://proceedings.mlr.press/v37/ganin15.html)
      3. Long at el. (2015) [Learning transferable features with deep adaptation networks](http://proceedings.mlr.press/v37/long15)

5. 방법 (how to enhance transferability guaranteeing acceptable discriminability)

   eigenvector corresponding to larger singular values should be leveraged for transferability

   eigenvector corresponding to smaller singular values should be leveraged for discriminability

   1. Batch Spectral Penalization

      suppress the dimension with top singular value to prevent it from standing out

      > F_s (feature from source), F_t (feature from target) 을 각각 SVD를 적용한다. 적용한 SVD에서 가장 큰 singular value들 중 몇개를 Loss로 사용한다.

      ![image-20210803163324942](pics/image-20210803163324942.png)

      

6. 실험

7. 유사 시도

8. 결론



### SHOT (2020)

1. SHOT의 주요 용어
   - UDA: Unsupervised Domain Adaptation



