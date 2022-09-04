# Bayesian Optimization

## Optimization

어떤 임의의 함수 f(x)-모델-의 값-모델의 점수-을 가장 크게(또는 작게)하는 해를 구하는 것이다. 이 모델은 수많은 변수를 가질 수 있다. 이 변수들을 조정하여 주어진 데이터 셋에 대해서 가장 좋은 결과를 내어놓는 x를 찾는 것이 hyperparameter 튜닝이다.





## 대표적인 hyperparameter 튜닝 기법들의 단점

### 1. GridSearch

- 모든 hyperparameter 후보들에 대한 일반화 성능을 확인하기 때문에, 시간이 너무 오래 걸린다.

### 2. RandomSearch

- 위에 비해 시간은 적게 걸리지만, 말 그대로 랜덤하게 뽑는 방식이기 때문에 정확도가 다소 떨어진다.



-> Bayesian Optimization은 이 둘에 비해 효율적으로 최적값을 찾아낸다는 장점이 있다.





## Bayesian Optimization

**이전의 정보를 최적 값 탐색에 반영하여 탐색을 반복해 최적의 값을 찾아 나가는 방식**



#### 1. Surrogate model

- 지금까지의 데이터를 통해 예상하는 모델을 만듦
- '최종 목적함수(우리가 찾으려고 하는)에 대해 확률적으로 추정한 결과'
- 이 모델을 기반으로 다음 탐색 지점을 결정
- (Gaussian Process를 주로 사용)

#### 2. Acquisition function

- 위 모델을 검증하고, 다음 모델을 만드는 데 있어서 최적의 하이퍼파라미터를 추천하는 단계
- (Expected Improvement 사용)



#### Bayesian Optimization 수행 과정

![img](https://blog.kakaocdn.net/dn/bfJ0Kh/btq6PDkXknB/TJHWGp8QCtKyD9vOhheBX1/img.png)

위의 파란색 선은 우리가 찾으려고 하는 목적함수를 나타내고, 검은색 점선은 지금까지 관측한 데이터를 바탕으로 우리가 예측한 estimated function을 의미한다. 검은색 점선 주변에 있는 파란 영역은, 목적함수 f(x)가 존재할만한 영역을 의미한다. 밑에 있는 EI(x)는 위에서 언급한 Acquisition function을 의미하며 다음 입력값 후보를 추천해준다. Acquisition function 값이 컸던 지점의 function 값을 관측하고 estimation을 update 한다.한다. 계속 update를 진행하면 estimation과 실제 function이 흡사해진다.

1 입력값, 목적 함수 및 그 외 설정값들을 정의합니다.

- 입력값 x : 여러가지 hyperparameter
- 목적 함수 f(x) : 설정한 입력값을 적용해 학습한, 딥러닝 모델의 성능 결과 수치(e.g. 정확도)
- 입력값 x 의 탐색 대상 구간 : (a,b)
- 입력값-함숫결과값 점들의 갯수 : n
- 조사할 입력값-함숫결과값 점들의 갯수 : N

2 설정한 탐색 대상 구간 (a,b) 내에서 처음 n 개의 입력값들을 랜덤하게 샘플링하여 선택합니다.

3 선택한 n 개의 입력값 x1, x2, ..., xn 을 각각 모델의 hyperparameter 로 설정하여 딥러닝 모델을 학습한 뒤, 학습이 완료된 모델의 성능 결과 수치를 계산합니다.

- 이들을 각각 함숫결과값 f(x1), f(x2), ..., f(xn) 으로 간주합니다.

4 입력값-함숫결과값 점들의 모음 (x1, f(x1)), (x2, f(x2)), ..., (xn, f(xn)) 에 대하여 Surrogate Model 로 확률적 추정을 수행합니다.

5 조사된 입력값-함숫결과값 점들이 총 N 개에 도달할 때까지, 아래의 과정을 반복적으로 수행합니다.

- 기존 입력값-함숫결과값 점들의 모음 (x1, f(x1)),(x2, f(x2)), ..., (xt, f(xt)) 에 대한 Surrogate Model 의 확률적 추정 결과를 바탕으로, 입력값 구간 (a,b) 내에서의 EI 의 값을 계산하고, 그 값이 가장 큰 점을 다음 입력값 후보 x1 로 선정합니다.
- 다음 입력값 후보 x1 를 hyperparameter 로 설정하여 딥러닝 모델을 학습한 뒤, 학습이 완료된 모델의 성능 결과 수치를 계산하고, 이를 f(x1) 값으로 간주합니다.
- 새로운 점 (x2, f(x2)) 을 기존 입력값-함숫결과값 점들의 모음에 추가하고, 갱신된 점들의 모음에 대하여 Surrogate Model 로 확률적 추정을 다시 수행합니다.

6 총 N 개의 입력값-함숫결과값 점들에 대하여 확률적으로 추정된 목적 함수 결과물을 바탕으로, 평균 함수 μ(x) 을 최대로 만드는 최적해를 최종 선택합니다. 추후 해당값을 hyperparameter 로 사용하여 딥러닝 모델을 학습하면, 일반화 성능이 극대화된 모델을 얻을 수 있습니다.



#### 실험 결과를 반영해가면서 효율적으로 hyperparameter을 찾아나가는 것이 가능해짐

![img](https://blog.kakaocdn.net/dn/Zs0RS/btq6QtI4Gu2/rxuwJbDeEsfqe0hihxtkz0/img.png)



#### 예제 코드

```python
# X에 학습할 데이터를, y에 목표 변수를 저장해주세요
X = train.drop(columns = ['index', 'quality'])
y = train['quality']
```

```python
# 랜덤포레스트의 하이퍼 파라미터의 범위를 dictionary 형태로 지정해주세요
## Key는 랜덤포레스트의 hyperparameter이름이고, value는 탐색할 범위 입니다.
rf_parameter_bounds = {
                      'max_depth' : (1,3), # 나무의 깊이
                      'n_estimators' : (30,100),
                      }
```

```python
# 함수를 만들어주겠습니다.
# 함수의 구성은 다음과 같습니다.
# 1. 함수에 들어가는 인자 = 위에서 만든 함수의 key값들
# 2. 함수 속 인자를 통해 받아와 새롭게 하이퍼파라미터 딕셔너리 생성
# 3. 그 딕셔너리를 바탕으로 모델 생성
# 4. train_test_split을 통해 데이터 train-valid 나누기
# 5 .모델 학습
# 6. 모델 성능 측정
# 7. 모델의 점수 반환

def rf_bo(max_depth, n_estimators):
  rf_params = {
              'max_depth' : int(round(max_depth)),
               'n_estimators' : int(round(n_estimators)),      
              }
  rf = RandomForestClassifier(**rf_params)

  X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size = 0.2, )

  rf.fit(X_train,y_train)
  score = accuracy_score(y_valid, rf.predict(X_valid))
  return score
```

```python
# 이제 Bayesian Optimization을 사용할 준비가 끝났습니다.
# "BO_rf"라는 변수에 Bayesian Optmization을 저장해보세요
BO_rf = BayesianOptimization(f = rf_bo, pbounds = rf_parameter_bounds,random_state = 0)

# pbounds: 테스트하고자 하는 hyperparameter의 집합
```

```python
# Bayesian Optimization을 실행해보세요
BO_rf.maximize(init_points = 5, n_iter = 5)

# init_points: random search로 탐색할 횟수
# n_iter: 최적값을 찾을 횟수(몇 개의 입력값 - 함숫값 점들을 확인할지)
# maximize: 목적함수를 최대로 만드는 최적해를 찾는다
```

```python
# 하이퍼파라미터의 결과값을 불러와 "max_params"라는 변수에 저장해보세요
max_params = BO_rf.max['params']

max_params['max_depth'] = int(max_params['max_depth'])
max_params['n_estimators'] = int(max_params['n_estimators'])
print(max_params)
```

```python
# Bayesian Optimization의 결과를 "BO_tuend_rf"라는 변수에 저장해보세요
BO_tuend_rf = RandomForestClassifier(**max_params)
```

