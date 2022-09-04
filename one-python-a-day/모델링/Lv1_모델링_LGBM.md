# 모델링



## train_test_split()

머신러닝 모델의 결과

![image-20220509193258492](Lv1_%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220509193258492-16520923797381.png)

머신러닝 모델에 train 데이터를 학습시킨 후 test데이터를 모델로 예측했을 경우 성능이 낮게 나오는 경우가 발생하는데 이를 overfitting 이라고 한다. 

overfit은 모델이 학습데이터에 지나치게 의존되어 있어서 다른 데이터(test)에 대해서는 예측율이 현저히 떨어지는 현상이다. 

이러한 현상을 방지하기 위해  train test로 구분되어 있는 데이터에서 train을 train과 validation으로 나눈 다음 학습 중간중간 validation데이터로 학습한 모델을 평가하여 모델이 학습된 정도를 평가하고 overfit이 발생하여 성능이 떨어지기 시작하면 학습을 종료시킨다. 

![image-20220509201840918](Lv1_%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220509201840918-16520951224432.png)

```python
# 라이브러리 로딩
from sklearn.model_selection import train_test_split

#train_test_split() 메소드를 이용해 train/validation 데이터 나누기 
x_train,x_valid, y_train, y_valid = train_test_split(train_x, train['category'])

# x_train,x_valid,y_train,y_valid 사이즈 확인

print('x_train 데이터 사이즈', x_train.shape)
print('x_valid 데이터 사이즈', x_valid.shape)
print('y_train 데이터 사이즈', y_train.shape)
print('y_valid 데이터 사이즈', y_valid.shape)
```



### test_size

- test data(validation data) 구성의 비율을 나타냅니다. train_size의 옵션과 반대 관계에 있는 옵션 값이며, 주로 test_size 파라미터를 지정 해줍니다. test_size = 0.2 로 지정 하면 전체 데이터 셋의 20%를 test(validation) 셋으로 지정하겠다는 의미입니다. default 값은 0.25 입니다.

  ```python
  x_train,x_valid, y_train, y_valid = train_test_split(train_x, train['category'], test_size=0.2)
  ```

  

### shuffle

- 데이터를 split 하기 이전에 섞을지 말지 여부에 대해 지정해주는 파라미터 입니다. default = True 입니다.

### stratify

- stratify값을 target 값으로 지정해주면 target의 class 비율을 유지 한 채로 데이터 셋을 split 하게 됩니다.

```python
x_train,x_valid, y_train, y_valid = train_test_split(train_x, train['category'], stratify=train['category'])
```



![image-20220509204421788](Lv1_%EB%AA%A8%EB%8D%B8%EB%A7%81.assets/image-20220509204421788-16520966629473.png)



## LGBM

```python
# LightGBM을 이용해 학습 및 검증 진행
from lightgbm import LGBMRegressor

model = LGBMRegressor()
model.fit(x_train,y_train, eval_set = [(x_train,y_train),(x_valid,y_valid)])
```



### eval_metric

- eval_metric = "원하는 평가산식" 을 정해주면 학습을 진행하며 지정한 평가 산식과 검증 데이터 셋을 이용해 결과 값을 출력

  '**l1**': absolute loss

  '**l2**': square loss

  '**rmse**': root square loss

  '**auc**': area under the ROC curve

  [참고문헌](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters)

  

### verbose 

- 일정한 값(숫자)을 지정해주면 n_estimators를 기준으로 지정해준 값마다 결과 값을 출력

```python
# LightGBM을 이용해 학습 및 검증 진행
from lightgbm import LGBMRegressor

#모델 정의 
model = LGBMRegressor()

#평가 산식을 AUC로 설정, n_estimators 기준 5번 마다 결과값 출력하게 모델 학습.
model.fit(train_x,train['category'], eval_set = [(x_train,y_train),(x_valid,y_valid)], eval_metric = 'auc' ,verbose = 5)
```





[참고문헌](https://deep-deep-deep.tistory.com/159)