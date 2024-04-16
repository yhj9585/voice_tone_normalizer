# voice_tone_normalizer

음성 소스를 받아 k-means 클러스터링으로 이를 몇 가지 군집으로 나누고
이를 각각 머신러닝을 이용하여 학습시켜 음성을 특정 연령이나 성별의 평균으로 맞추어줌

***

1. 음성 Data 전처리
   1) ~~데이터의 사람별로 데이터를 모아 MFCC를 이용하여 특징을 추출~~
   2) 나누어진 데이터를 군집으로 나눔 ?????
   3) 학습된 군집별로 각각 특정 인물이 어떤 군집인지 저장

3. 군집별로 각각 머신러닝을 이용해 학습
   군집의 평균적인 말투나 음색이 학습되어 그 군집의 다른 목소리를 입력해도 평준화할 수 있게 만듬

4. 새로운 데이터가 들어오면 클러스터링을 이용해 특정 군집으로 구분
   이후 학습된 각 머신러닝 데이터를 이용하여 평균치 목소리를 구함

***
