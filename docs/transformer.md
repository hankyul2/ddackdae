## Transformer

Transformer 논문을 읽으면서 중요한 점과 구현하면서 놓치기 쉬운 디테일한 부분을 정리했다.



이번에 Transformer를 구현하면서 느낀점은 모델보다도 Data가 훨씬 더 중요하다는 점이다. 모델은 조금 이상하면 정확도가 떨어질 뿐이지만 데이터가 이상하면 아예 학습이 되지 않았다.



### Data

- src : source text
- src_mask : source mask
- tgt_input : target input
- tgt_output : target output
- tgt_mask : target output mask



**교사 강요 학습**

source 부분은 크게 어려운 부분이 없다. 문제는 target 에서 일어났다. 트랜스포머 학습시에는 교사 강요를 통해서 학습 된다. 즉, target의 input은 translated 된 문장 중 맨 뒤에 단어를 제거한 버전이 사용된다. target의 output은 translated된 문장을 앞으로 한단어씩 땡긴 문장이 사용된다.

```python
## kr -> en
## en target input
target_input = ['Hi', 'My', 'name', 'is'] 
## en target output
target_output = ['My', 'name', 'is', 'hankyul']
```



**Mask**

[annotated transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) 를 보면 pad mask와 subsequence mask를 헷갈리게 만들었다. 분명 `torch.masked_fill()` 함수에 들어가면 오류가 날거 같은 모양으로 만들어놨다. 하지만 중간에 `unsqueeze()` 함수를 넣어서 내가 원래 생각했던 모양이랑 동일하게 만들어진다. 그래서 구현할 때 나는 처음 부터 `torch.masked_fill()` 함수에 들어갈 모양으로 만들었다.

```python
src = torch.randint(1, 10, size=(10, 10))
tgt = torch.randint(1, 10, size=(10, 10))
pad_mask = rearrange((src == pad_idx), 'b s -> b 1 1 s')
seq_mask = torch.triu(torch.ones((tgt.size(1), tgt.size(1))), diagonal=1) == 1
tgt_mask = pad_mask | seq_mask

## seq mask shape = (10, 10)
## pad mask shape = (10, 1, 1, 10)
## tgt mask shape = (10, 1, 10, 10)
```



*Tips: source mask와 target mask를 헷갈리지 말자!*



### Encoding/Decoding Layer

- src vocab size : source language의 vocab size
- tgt vocab size : target language의 vocab size



*Tips: source vocab size와 target vocab size를 필요이상으로 잡으면 그만큼 학습이 어려워진다*



### Positional Encoding

- div_term : positional encoding의 수식에 들어가는 한 부분



*Tips: position x div_term 에서 div_term을 계산할 때 항상 유의하자. 주의하지 않으면 학습이 되지 않는다*



### FeedForwardNetwork

- activation : weight 1과 weight 2 중간에 삽입되는 `relu` 혹은 `gelu` 같은 함수들



*Tips: activation을 빠뜨리지말고 넣어주자, 다만 작은 샘플에서는 큰 차이를 보이지 않는다*



### Encoder Layer

- sublayer connection index: norm layer가 중복되서 사용되는 경우가 있더라



*Tips: 큰 차이가 없다*



### Decoder Layer

- sublayer connection index: norm layer가 중복되서 사용되는 경우가 있더라



*Tips: 큰 차이가 없다*



### Optimizer

- factor : 최대 learning rate의 크기가 결정되는 계수이다.
- warmup : learning rate의 최대치를 결정하는 계수이다. 



*Tips: warmup을 하지 않을 경우 모델이 어렵다*



### Label Smoothing

- log_softmax : KLDivLoss를 계산하기 위해서는 직접 `log_softmax` 함수를 호출해야한다.
- smoothness : label smoothing에서 핵심적인 값이다.
- tokens : tokens로 loss 값을 나눠준다.



*Tips: smoothing은 작은 데이터셋에서는 큰 효과를 발휘하지 않는다, CrossEntropy를 써도 크게 상관은 없다. 정확도가 조금 떨어질 뿐이다.*

*Tips: log_softmax를 하지 않을 경우 KLDivLoss의 값이 음수로 나오고 절대값이 매우 크게 나온다*

*Tips: annotated transformer와 loss를 똑같이 내기 위해서는 `KLDivLoss(reduction='sum')`을 사용해야 한다. 나도 왜그런지는 잘 모르겠다*



