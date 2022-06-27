# KBO 타자 OPS 예측 경진대회_모델링

## 4. Model Building

```python
# 출력 할 때 마다, 기존 출력물들은 제거해주는 모듈
from IPython.display import clear_output

# 모델 모듈
from xgboost import XGBRegressor, plot_importance
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = result.fillna(-1) # nan값 -1로 impute 

df = df[df['OPS_NEXT'] &gt; 0] 
# 다음 해의 OPS가 존재하는 값들만 추출 (다음 해 OPS 가 없다면, predict 값과 target값을 비교 할 수 없기 때문)
df = pd.concat([df, pd.get_dummies(df['포지션'], drop_first=True)], axis=1) # 포지션 dummy variable로 변환
```

### 학습

```python
# 사용 features 명시
infos = ['ID','이름','생일','팀','시즌','포지션','나이']
stats = ['G','타수','득점','안타','1타','2타','3타','홈런','루타','타점','도루','도실','볼넷','사구','고4','삼진'\
        ,'병살','희타','희비','타율','출루','장타','OPS']


# 누적 및 lag stat features 이름
stats_cum = [i + '_누적' for i in stats]
stats_lag1 = [i + '_LAG1' for i in stats]
stats_lag2 = [i + '_LAG2' for i in stats]
stats_lag3 = [i + '_LAG3' for i in stats]


# X features와 y feature 정리
stats_position = ['2B', '3B', 'C', 'CF', 'DH', 'LF', 'RF', 'SS']
X_cols = stats + stats_cum + stats_lag1 + stats_lag2 + stats_position + ['나이']
y_cols = ['OPS_NEXT']


# 학습시킬 모델
model_dict = {'xgb':XGBRegressor(n_estimators=110, learning_rate=0.05 ,max_depth=3, min_samples_split=400, random_state=23),
              'lgbm':LGBMRegressor(),
              'rf':RandomForestRegressor(),
              'svr':SVR(),
              'knn':KNeighborsRegressor(),
              'reg':LinearRegression(),
              'ridge':Ridge(),
              'lasso':Lasso()}

# 2009 ~ 2016년 기간의 데이터로 검증
# 예를들어 2010년은 2009년까지의 데이터로, 2011년은 2010년까지의 데이터로 검증
# 에러가 가장 낮은 두 모델, xgboost와 ridge 중 ridge 선택
# ridge가 xgboost에 비해 과적합이 적고 일반화가 더 잘 이뤄졌을 것이라는 판단

test_error = []
r2 = []

for year in range(2010, 2018):

    train = df[df['시즌'] &lt; year-1].reset_index(drop=True)
    test = df[df['시즌'] == year-1].reset_index(drop=True)
    
    X_train = train[X_cols]
    y_train = train[y_cols]

    X_test = test[X_cols]
    y_test = test[y_cols]
    
    model = model_dict['ridge']    #모델명 바꾸어 가면서 실험 가능
    weight = train['타수']
    model.fit(X_train, y_train, sample_weight=weight)
    ## sample_weight: 샘플 가중치 (왜 사용하나??)
    y_predict = model.predict(X_test)
    
    test_error.append(mean_squared_error(y_test, y_predict, sample_weight=X_test['타수']) ** 0.5)
    r2.append(r2_score(y_test, y_predict))
    ## r2: 결정계수. 내가 만든 이 모델이 타겟 변수를 얼마나 잘 예측, 설명하는지 평가하는 지표 중 하나.

    print(year, ": 완료")
    
    
print("test error : ", np.mean(test_error))
print("test std : ", np.std(test_error))  # 표준편차 계산
print("r2 : ", np.mean(r2))
print("완료")

2017 : 완료
test error :  0.12297661095675332
test std :  0.017695723649870054
r2 :  0.1972005395338279
완료

# 연도별 테스트 에러
test_error
[0.11965635488780504,
 0.10017133715584686,
 0.1113487547340026,
 0.11995787803170647,
 0.1585607101594155,
 0.12152418699996415,
 0.14271784504461912,
 0.10987582064066678]

# 예측값과 실제값의 시각화
plt.scatter(y_test, y_predict)
plt.xlabel("실제 값")
plt.ylabel("예측 값")
plt.xlim(0,1.5)
plt.ylim(0,1.5)
(0, 1.5)
```

![img](data:image/png;base64,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)

## 제출

```python
predict = result[result['시즌']==2018] # 2018년도 데이터 프레임
predict = pd.concat([predict, pd.get_dummies(predict['포지션'], drop_first=True)], axis=1) 

# 포지션 dummy variable로 변환
## get_dummies: ML을 할 때 기계가 이해할 수 있도록 모든 데이터를 수치로 변환해주는 전처리 작업이 필수적 https://devuna.tistory.com/67

# Dacon regular season과 submission 병합

# 추후 statiz data와 병합할 때 기준이 되는 생일 데이터를 얻기 위함
reg_sub = pd.merge(submission, reg[reg['year']==2018], on='batter_id', how='left', suffixes=['','_reg'])
## how='': 어느 데이터프레임 기준으로 병합할 것인가 (https://mizykk.tistory.com/82)

# regular season dataframe의 생일 데이터를 statiz와 일치시킴
reg_sub['year_born'] = reg_sub['year_born'].apply(lambda x: datetime.strptime(x, "%Y년 %m월 %d일").date())
## datetime.striptime: 날짜와 시간 형태의 문자열을 datetime 형식으로 변환해줌

reg_sub['year_born'] = reg_sub['year_born'].astype(str)

# 필요한 feature만 남김
reg_sub = reg_sub[['batter_name', 'year_born', 'batter_id']]

# 이름과 생일을 기준으로 dacon data와 statiz data 병합
total = pd.merge(reg_sub, predict, how='left', left_on=['batter_name', 'year_born'], right_on=['이름', '생일'])
X = total[X_cols] # 사용 feature
X = X.fillna(-1) # NA값 -1로 impute

# 예측
total['y_hat'] = model.predict(X)

# 타자별 예측값 feature에 저장
submission['batter_ops'] = submission['batter_id'].apply(lambda x: total[total['batter_id']==x]['y_hat'].values[0])
## https://koreadatascientist.tistory.com/115

# 제출 파일 생성
submission.to_csv("data/tnt_submission.csv", index=False, encoding='utf-8')

# 제출 결과 확인
sub = pd.read_csv('data/tnt_submission.csv')
sub
```

|      | batter_id | batter_name | batter_ops |
| ---: | --------: | ----------: | ---------: |
|    0 |         1 |      강경학 |   0.658489 |
|    1 |         2 |      강구성 |   0.546550 |
|    2 |         3 |      강민국 |   0.592865 |
|    3 |         4 |      강민호 |   0.852663 |
|    4 |         5 |      강백호 |   0.894065 |
|    5 |         8 |      강상원 |   0.596299 |
|    6 |         9 |      강승호 |   0.600272 |
|    7 |        11 |      강진성 |   0.595175 |
|    8 |        12 |      강한울 |   0.601566 |
|    9 |        16 |      고명성 |   0.588540 |
|   10 |        18 |      고종욱 |   0.752714 |
|   11 |        19 |      구자욱 |   0.881511 |
|   12 |        20 |      국해성 |   0.640459 |
|   13 |        22 |      권정웅 |   0.606158 |
|   14 |        23 |      권희동 |   0.758239 |
|   15 |        24 |      김강민 |   0.738815 |
|   16 |        28 |      김규민 |   0.681435 |
|   17 |       101 |      문선재 |   0.596781 |
|   18 |        30 |      김동엽 |   0.815607 |
|   19 |        33 |      김동한 |   0.603093 |
|   20 |        35 |      김문호 |   0.708531 |
|   21 |        36 |      김민성 |   0.766009 |
|   22 |        37 |      김민수 |   0.532364 |
|   23 |        38 |      김민식 |   0.643187 |
|   24 |       335 |      홍성갑 |   0.544609 |
|   25 |        39 |      김민하 |   0.671040 |
|   26 |        40 |      김사훈 |   0.531915 |
|   27 |        41 |      김상수 |   0.719601 |
|   28 |        43 |      김선빈 |   0.796683 |
|   29 |        44 |      김성욱 |   0.702760 |
|  ... |       ... |         ... |        ... |
|  190 |       294 |      지석훈 |   0.558775 |
|  191 |       295 |      지성준 |   0.659402 |
|  192 |       300 |      채은성 |   0.917660 |
|  193 |       301 |      채태인 |   0.764336 |
|  194 |       307 |      최승준 |   0.706538 |
|  195 |       308 |      최영진 |   0.690007 |
|  196 |       309 |      최원제 |   0.678761 |
|  197 |       310 |      최원준 |   0.713414 |
|  198 |       311 |      최윤석 |   0.553565 |
|  199 |       312 |      최재훈 |   0.647541 |
|  200 |       313 |        최정 |   0.944987 |
|  201 |       314 |      최정민 |   0.657098 |
|  202 |       315 |      최주환 |   0.876702 |
|  203 |       316 |      최준석 |   0.670725 |
|  204 |       317 |      최진행 |   0.737720 |
|  205 |       318 |        최항 |   0.714403 |
|  206 |       319 |      최형우 |   0.993787 |
|  207 |       324 |      하주석 |   0.746670 |
|  208 |       325 |      하준호 |   0.663997 |
|  209 |       326 |      한동민 |   0.938371 |
|  210 |       327 |      한동희 |   0.678485 |
|  211 |       329 |      한승택 |   0.641606 |
|  212 |       331 |      허경민 |   0.826822 |
|  213 |       332 |      허도환 |   0.641374 |
|  214 |       334 |        호잉 |   0.890421 |
|  215 |       337 |      홍재호 |   0.593042 |
|  216 |       338 |      홍창기 |   0.643868 |
|  217 |       341 |      황윤호 |   0.565175 |
|  218 |       342 |      황재균 |   0.887563 |
|  219 |       344 |      황진수 |   0.625428 |

220 rows × 3 columns