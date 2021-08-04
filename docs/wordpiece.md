## WordPiece

WordPiece 모델은 ["japanese and korean voice search(2012)"](https://ieeexplore.ieee.org/abstract/document/6289079) 라는 논문에서 처음 제안 되었다. 해당 논문을 읽는데 너무너무너무 어려웠다. voice recognition을 몰라서 그런지 몰라도 내가 지금 읽고 있는 부분에서 말하는게 Speech Dictionary를 만하는 것인지 Written Dictionary를 말하고 있는건지 읽는 내내 헷갈렸다. 그래도 WordPiece 모델이 뭔지는 확실히 알게되었다. 이 논문은 내가 아직 다 읽었다고 할 수 없다. 나중에 voice recognition을 제대로 공부할 때 다시 읽어보고 이번에는 wordpiece에 대한 소개를 간략히 정리한다.

다음시간에 할 것 : n-gram, perplexity 구현해보기, [WordPiece모델 구현 코드 확인하기](https://lovit.github.io/nlp/2018/04/02/wpm/)



### 요약

문제 : 일본어와 한국어를 이용한 음성 검색 시스템을 만들때 발생한 문제와 해결책을 소개함.

접근법 : Written Language Domain에서 Language Model과 Dictionary를 완벽하게 만드는것이 모델의 복잡도를 줄여줌

해결책 : 어떻게 이런 Dictionary와 Language Model을 만들 수 있는지 소개할께^^



### 또입

많은 아시아 국가(한국, 일본, 중국)들의 기본 단어이 매우 많고 복잡하다. 따라서 아시안 국가의 발음 사전을 만드는 것은 매우 어렵다. 따라서 이런 기본 단어들을 분리하기 위한 segmenter를 만들었다. 이 segmenter들의 경우 특별한 전처리 없이 어떤 언어들에서도 사용할 수 있다.



### 음성 데이터 수집

구글이 구글해서 음성 데이터를 수집했다.



### Segmentation and Word Inventory

1. 문장들을 전부 글자 단위로 쪼갠다(가, 나, 다, 라, ... 핳)
2. 말뭉치(쪼개진 문장들)를 통해서 모델을 학습한다.
3. 학습한 모델의 확률을 가장 높일 수 있는 단어쌍을 선택한다.
4. 2~3번을 계속해서 반복한다. 언제까지?? 정해진 단어의 개수에 도달하거나 증가하는 확률이 더 이상 못봐줄꺼 같을때 까지 



### 꼭 알아야 하는 것 정리

참고 자료 

- [김기현의 자연어 처리 캠프](https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-8/03-perpexity)  
- [딥러닝을 이용한 자연어처리 입문](https://wikidocs.net/31695)



1. Language Model (LM) : 단어의 확률을 부여하는 모델. 옛날에는 통계 기반의 n-gram 방식을 사용했지만 최근에는 Deep-learning 모델을 사용하는 편

2. Perplexity (PPL) : 피피엘(PPL) 하니깐 뭔가 광고와 관련된 단어 같지만 사실 Perplexity(PPL)의 줄임말이다. 수식 자체는 복잡하지 않지만 이를 우리가 사용하는 식으로 유도 하는 것은 굉~~장히 어려운 일이기 때문에 (대학원가서 해봐야징~) PPL 의 원래 수식과 우리가 사용할 수 있는 식으로 최종 변형된 식만 이해하면 된다. 당연한 말이겠지만 dictionary의 size가 늘어나면 perplexity가 증가한다. 하지만 perplexity는 작을 수록 좋은 것이다. 그렇자면 perplexity를 줄이기 위해서 dictionary의 사이즈를 줄이는 것이 좋을까???

   

   <img src="https://bit.ly/3A9kDPG" align="center" border="0" alt="PPL(w_{1},w_{2},\ldots,w_{n})=P(w_{1},w_{2},\ldots,w_{n})^{-\frac{1}{n}}" width="357" height="26" />

   

   <img src="https://bit.ly/3lsC7Cx" align="center" border="0" alt="\text{PPL}=\exp(\text{Cross Entropy})" width="214" height="19" />

3. n-gram (SLM) :

4. BLEU : modified n-gram precision 의 weigthed log probability 의 exp 값을 의미한다. 각각의 식으로 나타내면 다음과 같다.

   각각의 의미를 살펴보면 Count를 사용하는 이유는 예측 문장의 단어가 실제 문장에서 얼마나 등장하는지를 알아보기 위한 것이다. 그리고 clip을 사용하는 이유는 중복된 단어가 많이 나오는것을 방지하기 위함이다. 그리고 n-gram을 사용하는 이유는 순서와 길이에 대한 페널티를 주기 위함이다. 마지막으로 BP 값은 짧은 번역문장에 대한 페털티를 주기 위한 것이다. 

   <img src="https://bit.ly/37j58Zc" align="center" border="0" alt="Count(n-gram) = frequency" width="260" height="19" />

   

   <img src="https://bit.ly/37j5a3g" align="center" border="0" alt="Count_{clip}(n-gram) = min(Count, Max\_ref\_count)" width="428" height="21" />

   

   <img src="https://bit.ly/3xis8lv" align="center" border="0" alt="p_{n} =  \frac{ \sum_{n-gram \in candidate}^{} Count_{clip}(n-gram)}{\sum_{n-gram \in candidate}^{} Count(n-gram)" width="360" height="58" />

   

   <img src="https://bit.ly/3s4E89z" align="center" border="0" alt="BLEU = BP \times \exp( \sum_{n}^{N} w_{n} \times \log(p_{n}) )" width="287" height="50" />

   이를 구현한 [nltk라이브러리](https://www.nltk.org/api/nltk.translate.html)를 사용할 수 있다. 

   ```python
   import nltk.translate.bleu_score as BLEU
   
   hypothesis = 'this is korea'
   references = ['hi this is korea', 'This is seoul, korea', 'Welcome to Korea']
   
   chencherry = BLEU.SmoothingFunction()
   BLEU_score = BLEU.sentence_bleu(list(map(lambda x: x.split(), references)) , hypothesis.split(), weights=(0.25, 0.5, 0.24, 0.1), smoothing_function=chencherry.method1)
   
   print(BLEU_score)
   ```

   



