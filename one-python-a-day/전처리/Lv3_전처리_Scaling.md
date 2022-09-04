# 전처리- 이상치

## 이상치 

> 패턴에서 벗어난 값, 중심에서 많이 벗어난 값

- 평균에 막대한 영향을 미침
- [1,2,3,4,5] 의 평균은 3이지만, [1,2,3,4,100]의 평균은 22  => 하지만 중위수는 3으로 같음
- 따라서 평균으로 값을 나타내기보다 중위수로 요약값을 나타내는 것이 더 결과를 잘 반영하는 경우도 있음
- 이상치 데이터는 모델의 성능을 크게 떨어트림



## 이상치 탐지 방법 - IQR

> IQR이란, Interquartile range의 약자로써 Q3 - Q1를 의미한다.
>
> Q3 - Q1: 사분위수의 상위 75% 지점의 값과 하위 25% 지점의 값 차이

![img](%EC%A0%84%EC%B2%98%EB%A6%AC(Lv3).assets/img-16508985826691.png)

Q1 -  1.5 * IQR : 최소 제한선, Q3 + 1.5 * IQR :최대 제한선

 최소 제한선과 최대 제한선을 넘어가는 값들을 이상치라고 한다. 

```python
from collections import Counter 
def detect_outliers(df, n, features): 
    outlier_indices = [] 
    for col in features: 
        Q1 = np.percentile(df[col], 25) 
        Q3 = np.percentile(df[col], 75) 
        IQR = Q3 - Q1 
        outlier_step = 1.5 * IQR 
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index 
        outlier_indices.extend(outlier_list_col) 
    outlier_indices = Counter(outlier_indices) 
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
return multiple_outliers 

Outliers_to_drop = detect_outliers(df_train, 2, ["변수명"])
```

출처: https://hong-yp-ml-records.tistory.com/15 [HONG YP's Data Science BLOG]







## 수치형 데이터 정규화

> 의사결정 나무나, 랜덤포레스트 같은 “트리 기반의 모델”들은 대소 비교를 통해서 구분하기 때문에, 숫자의 단위에 크게 영향을 받지 않습니다.
>
> 하지만 Logistic Regression, Lasso 등과 같은 “평활 함수 모델”들은 숫자의 크기와 단위에 영향을 많이 받습니다.   출처: 데이콘

[데이터 평활](https://min23th.tistory.com/20) : 시계열 데이터에서 발생하는 무작위적인 변화량으로 인해 생기는 효과를 줄이기 위해 사용되는 기법 

[Logistic Regression](https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1_%ED%9A%8C%EA%B7%80) :  독립 변수의 선형 결합을 이용하여 사건의 발생 가능성을 예측하는데 사용되는 통계 기법이다.



### Scaler

- StandardScaler : 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다. 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

```python
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
print(standardScaler.fit(train_data))
train_data_standardScaled = standardScaler.transform(train_data)
```

- MaxAbsScaler : 절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

```python
from sklearn.preprocessing import MaxAbsScaler
maxAbsScaler = MaxAbsScaler()
print(maxAbsScaler.fit(train_data))
train_data_maxAbsScaled = maxAbsScaler.transform(train_data)
```

- RobustScaler : 아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포시킨다. 

```python
from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
print(robustScaler.fit(train_data))
train_data_robustScaled = robustScaler.transform(train_data)
```

[출처](https://mkjjo.github.io/python/2019/01/10/scaler.html)



### Min Max Scailing 

> 변수의 크기가 상대적으로 값이 너무 작거나, 큰 경우 해당 변수가 Target에 미치는 영향력이 제대로 표현되지 않을 수 있음
>
> 이러한 문제를 해결하기 위해 변수의 값의 범위를 0부터 1로 변경하고 최솟값을 0, 최댓값을 1로 해서 값들의 분포를 나타내는 것
>
> 이때 이상치에 따라 scailing의 값이 변하므로 이상치에 매우 민감하고, 이상치를 처리해주는 것이 매우 중요
>
> 이상치가 있는 경우 변환된 값이 매우 좁은 범위에 몰려있을 수 있음

- 공식 : 

![image-20220425205122523](%EC%A0%84%EC%B2%98%EB%A6%AC(Lv3).assets/image-20220425205122523-16508986033022.png)





- 예제 코드:

```python
from sklearn.preprocessing import MinMaxScaler

# scaler로 MinMaxScaler()함수를 불러온다.
scaler = MinMaxScaler()

# scaler를 학습시켜야함
# train데이터에 []을 더 붙여주는 이유는 scaler.fit의 인자는 2d array로 받기 때문이다.
scaler.fit(train[['fixed acidity']])

# 학습된 scaler로 새로운 컬럼을 만든다.
train['Scaled fixed acidity'] = scaler.transform(train[['fixed acidity']])



# 위의 방법을 fit_transform으로 한꺼번에 써줄 수 있다.


train['스케일된 컬럼명']=scaler.fit_transform(train['컬럼명']) #에러 발생 
'''
ValueError: Expected 2D array, got 1D array instead:
array=[5.6 8.8 7.9 ... 7.8 6.6 7. ].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
'''
# 이 오류는 2d array를 인자로 받아야 하는데, 리스트는 1d array이기 때문에 오류가 난다.
# 해결책은 안에 []를 더 넣어 붙인다!


train['Scaled 컬럼명']=scaler.fit_transform(train[['컬럼명']])
```







## One-Hot Encoding

> 컴퓨터는 문자 데이터를 학습할 수 없으므로 문자로 되어있는 feature를 컴퓨터가 읽어서 학습할 수 있도록 인코딩해야 한다. 



### 단어 집합 

- 서로 다른 단어들의 집합 
- 기본적으로 단어의 변형 형태도 다른 단어로 간주한다. (book, books 서로 다른 단어)



### 원-핫 인코딩

- 단어 집합의 크기만큼 각 단어별로 고유한 정수를 부여한다. (각 단어의 인덱스가 됨)
- 표현하고 싶은 단어의 인덱스에 1을 부여하고, 다른 인덱스에는 0을 부여하는 벡터 표현 방식

```python
[[0. 0. 1. 0. 0. 0. 0. 0.] # 인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] # 인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] # 인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] # 인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] # 인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] # 인덱스 7의 원-핫 벡터

[[1,0]
[0,1]]
```



### 단점

- 단어의 개수가 늘어날수록 이를 벡터로 표현하기 위해 필요한 저장 공간이 증가하게된다.  또한 벡터로 저장할 때 하나만 1이 부여되고 나머지는 0이 부여되기 때문에 부여된 저장 공간을 비효율적으로 사용한다. 
- 단어의 유사도를 표현하지 못함



[출처](https://wikidocs.net/22647) 
