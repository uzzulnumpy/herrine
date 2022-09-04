# 모델링

## GBM 

- 부스팅 알고리즘은 여러 개의 weak learner를 순차적으로 학습하면서 각 단계별로 잘못 예측된 데이터에 대해 가중치를 부여해 오류를 개선해 나가는 학습 방식

![img](%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/img-16514852894803.png)

- 각 **Iteration(step)**에서 잘못 예측된 데이터에 대하여 가중치를 부여한 예측 결정 기준을 결합해 최종 예측 결정 기준을 만듬
- GBM(Gradient Boost Machine)도 AdaBoost와 유사하게 동작하지만 가중치 업데이트의 방식이 **경사 하강법(Gradient descent)**을 사용하는 것이 차이점
- GBM은 RandomForest보다 나은 예측 성능을 보이는 경우가 많지만 수행 시간이 오래 걸린다는 단점이 있음 빠른 수행 시간을 요구하는 경우에는 RandomForest를 사용

### GBM 하이퍼 파라미터 튜닝

- loss : 경사 하강법에서 사용할 비용 함수를 지정. 기본값은 'devidence'
- learning_rate : GBM이 학습을 진행할 때마다 적용하는 학습률 기본값(0.1),   weak learner가 순차적으로 오류 값을 보정해 나가는 데 적용하는 계수로 0 ~ 1 사이의 값, 
- n_estimators : weak learner의 개
- subsample : weak learner가 학습에 사용하는 데이터 샘플링의 비율, 기본값은 1.

[참고문헌](https://kimdingko-world.tistory.com/181)



## XGBoost

### Boosting

- 여러 개의 성능이 높지 않은 모델을 조합해서 사용하는 Ensemble 기법 중 하나
- 약한 예측 모형들의 학습 에러에 가중치를 두고, 순차적으로 다음 학습 모델에 반영하여 강한 예측모형을 만드는 것이다.

![image-20220502182813214](%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220502182813214-16514836948861.png)

### XGBoost 

- XGBoost는 Extreme Gradient Boosting의 약자
- Boosting 기법을 이용하여 구현한 알고리즘은 Gradient Boost 가 대표적인데 이 알고리즘을 병렬 학습이 지원되도록 구현한 라이브러리가 XGBoost 
- Regression, Classification 문제를 모두 지원하며, 성능과 자원 효율이 좋아서, 인기 있게 사용되는 알고리즘

### XGBoost의 장점

- 기존 모델(GBM) 대비 빠른 수행시간
  - 병렬 처리로 학습, 분류 속도가 빠르다.
- 과적합 규제(Regularization)
  - 표준 GBM의 경우 과적합 규제기능이 없으나, XGBoost는 자체에 과적합 규제 기능으로 강한 내구성 지닌다.
- 분류와 회귀영역에서 뛰어난 예측 성능 발휘
- Early Stopping(조기 종료) 기능이 있음
- 다양한 옵션을 제공하며 Customizing이 용이
- 결측치를 내부적으로 처리

[참고문헌](https://wooono.tistory.com/97)

[참고문헌](https://xgboost.readthedocs.io/en/latest/python/python_api.html)





## LGBM

- Tree 기반 학습 알고리즘
- 기존의 다른 Tree 기반 학습 알고리즘과 달리 Tree 구조가 수평적으로 확장하지 않고 수직적으로 확장함

![image-20220502192706129](%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220502192706129-16514872272064.png)

- 장점 : 
  - 대용량 데이터 처리
  - 효율적인 메모리 사용
  - 빠른 속도
  - GPU 지원

- 단점 : 과적합 우려가 다른 Tree 알고리즘 대비 높아서 데이터의 양이 적을 경우 사용 자제

[참고문헌](https://nicola-ml.tistory.com/51)





## stratified k-fold

- 데이터가 편향되어 있을 경우 k-fold 교차검증을 사용하면 성능 평가가 잘 되지 않을 수 있음, 이를 해결하기 위해 stratified k-fold 방법을 사용

### ![image-20220502205115616](%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220502205115616-16514922766947.png)

[참고문헌](https://jinnyjinny.github.io/deep%20learning/2020/04/02/Kfold/)



### Voting Classifier

- 여러 개의 모델을 결합하여 예측 결과를 도출하는 앙상블 기법 

### Hard Voting

- 여러 모델을 생성하고 그 결과들을 집계하여 가장 많은 표를 얻는 클래스를 최종 예측값으로 정하는 것

![image-20220502204434249](%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220502204434249-16514918755535.png)

### Soft Voting 

- 앙상블에 사용되는 모든 분류기가 클래스의 확률을 예측할 수 있을 때 사용 가능
- 각 분류기의 예측을 평균해서 확률이 가장 높은 클래스로 예측

![image-20220502204626358](%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220502204626358-16514919876396.png)

[참고문헌](https://nonmeyet.tistory.com/entry/Python-Voting-Classifiers%EB%8B%A4%EC%88%98%EA%B2%B0-%EB%B6%84%EB%A5%98%EC%9D%98-%EC%A0%95%EC%9D%98%EC%99%80-%EA%B5%AC%ED%98%84)