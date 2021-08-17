## Neural Machine Translation of Rare Words with Subword Units (BPE)

Rico Sennrich et al. 2016



배경 (그냥 내가 간단하게 정리)

1. 2012년도에 N-gram 기반의 Probability를 높일 수 있는 WordPiece 모델이 제시되었다.
2. 4년뒤인 2016년도에 BPE(지금 이 논문)과 WordPiece를 이용한 Translation System 논문이 공개되었다.



### 요약

문제 : Neural Machine Translation(NMT)할 때 Out-of-Vocabulary 문제가 발생한다.

제안 방법 : Unknown word를 Sequence of Subword units 으로 Encoding해서 표현 한다. (BPE)

결과 : 가장 간단한 N-gram 기반으로 했을 때와 BPE로 했을 때의 Translation Task의 성능이 1.3 ~ 1.1 BELU Score 만큼 오른다.



### 1. 도입

문제 : 복합어, 외래어, 지명 같은 단어를 Translation 하기 어렵다.

방법 : Subword를 사용하면 open-vocabulary neural machine translation을 할 수 있다.(BPE)



### 2. Neural Machine Translation

모델의 성능 측정은 그 당시 널리 사용되었던 Bahdanau의 Encoder-Decoder + Attention 모델이다. 흠... 이걸 직접 구현할지 아니면 그냥 N-gram 기반의 성능 측정을 할 지 모르겠다.



### 3. Not Yet



