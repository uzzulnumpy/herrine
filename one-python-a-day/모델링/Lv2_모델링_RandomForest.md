# RandomForest & 'criterion' 옵션

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
```



#### RandomForest란?

https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/

여러 개의 의사결정나무를 만들어서 이들의 평균으로 예측의 성능을 높이는 방법이며, 이러한 기법을 앙상블(Ensemble) 기법이라고 한다. 

주어진 하나의 데이터로부터 여러 개의 랜덤 데이터셋을 추출해, 각 데이터셋을 통해 모델을 여러 개 만들 수 있다.

의사 결정 나무는 가지치기(오차를 크게 할 위험이 높거나 부적절한 추론규칙을 가지고 있는 가지 또는 불필요한 가지를 제거하는 단계)를 함에도 불구하고 overfitting(학습 데이터를 과하게 학습해 실제 데이터에서는 정확도가 떨어지는 현상)되는 경향이 있어 일반화 성능이 좋지 않다.

이를 보완하기 위해 앙상블 기법을 사용하는 것이다.



![ensemble model random forest](RandomForest.assets/325745-Bagging-ensemble-method.png)

*bootstrap: 표본에서 재표본을 여러번 추출하여 표본에 대해 더 자세히 알고자 하는 데 사용됨

![인스턴스 랜덤 포레스트](RandomForest.assets/33019random-forest-algorithm2.png)



## RandomForestRegressor()의 criterion 옵션

"랜덤포레스트 모듈의 옵션 중 criterion 옵션을 통해 어떤 평가척도를 기준으로 훈련할 것인지 정할 수 있습니다. 

따릉이 대회의 평가지표는 RMSE 입니다. RMSE 는 MSE 평가지표에 루트를 씌운 것으로서, 모델을 선언할 때 criterion = ‘mse’ 옵션으로 구현할 수 있습니다."



### RandomForestRegressor()

랜덤포레스트 회귀 분석



### model = RandomForestRegressor(criterion = 'mse')

- RMSE(Root Mean Squared Error): 예측값과 실제값을 뺀 후 제곱시킨 값들을 모두 더하고 n으로 나눈 후 루트를 씌운다. 쉽게 말해 오차의 제곱에 대한 평균으로, 이 값이 작을 수록 원본과의 오차가 적은 것이므로 추측한 값의 정확성이 높은 것이다. (criterion의 디폴트 값으로, "squared_error"로 용어 변경됨.)

## RandomForestClassfier

### 사이킷런 모델

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

- **n_estimators** : 모델에서 사용할 트리 갯수(학습시 생성할 트리 갯수)
- **criterion** : 분할 품질을 측정하는 기능 (default : gini)
  - [지니와 엔트로피](https://wyatt37.tistory.com/9) 
  - 결론 
    - 시간을 투자해서 더 나은 성능을 원하면 엔트로피
    - 준수한 성능과 빠른 계산을 원하면 지니
    - 그떄마다 좋은 성능을 나타내는 것이 다름
- **max_depth** : 트리의 최대 깊이
- min_samples_split : 내부 노드를 분할하는데 필요한 최소 샘플 수 (default : 2)
- min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1)
- min_weight_fraction_leaf : min_sample_leaf와 같지만 가중치가 부여된 샘플 수에서의 비율
- **max_features** : 각 노드에서 분할에 사용할 특징의 최대 수
- max_leaf_nodes : 리프 노드의 최대수
- min_impurity_decrease : 최소 불순도
- min_impurity_split : 나무 성장을 멈추기 위한 임계치
- **bootstrap** : 부트스트랩(중복허용 샘플링) 사용 여부
- oob_score : 일반화 정확도를 줄이기 위해 밖의 샘플 사용 여부
  - Out-Of-Bag (OOB)
  - 일반적으로 훈련 샘플의 63%만 샘플링 되고 나머지 37%의 데이터는 훈련에 쓰이지 않는다.
- **n_jobs** :적합성과 예측성을 위해 병렬로 실행할 작업 수
- **random_state** : 난수 seed 설정
  - [random_state](https://miinkang.tistory.com/19)
  - **데이터를 섞을때 일관적이게 섞고 싶을 때 사용하는 것**이 seed
- verbose : 실행 과정 출력 여부
- warm_start : 이전 호출의 솔루션을 재사용하여 합계에 더 많은 견적가를 추가
- class_weight : 클래스 가중치