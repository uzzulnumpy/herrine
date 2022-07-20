# 2020 D CUP Google Analytics 데이터 경진대회_2

> Private 2위, Private3.28814점, LSTM+XGB+Extra Tree



# [ 목차 ]

1. 라이브러리 및 데이터 불러오기
2. 추가 데이터 처리
3. EDA
4. LSTM
5. XGBoost
6. Extra Tree
7. Weighted average Ensemble

# 1. 라이브러리 및 데이터 불러오기

```python
import pandas as pd
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas_datareader.data as pdr
import numpy as np
# xgboost: 여러개의 약한 Decision Tree를 조합해 사용하는 앙상블 기법 중 하나.
# https://wooono.tistory.com/97
from xgboost import plot_importance, plot_tree
from numpy import asarray
from math import sqrt
from matplotlib import pyplot
from datetime import date

import warnings
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import chart_studio.plotly as py
import plotly.offline as pyo
import plotly.io as pio
from plotly.subplots import make_subplots
import matplotlib as mpl   
import matplotlib.font_manager as fm
import seaborn as sns
plt.style.use('ggplot')
%matplotlib inline
plt.rcParams['font.family']='NanumGothic'
pio.renderers.default = 'notebook_connected'
print(plt.rcParams['font.family'])

from xgboost import XGBRegressor
# ExtraTreesRegressor: 더욱 랜덤한 결정 트리 앙상블 기법. 과적합을 막아주고 모델의 일반화 성능이 더 높음.
# https://tensorflow.blog/2017/11/30/%EB%8D%94%EC%9A%B1-%EB%9E%9C%EB%8D%A4%ED%95%9C-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8-%EC%9D%B5%EC%8A%A4%ED%8A%B8%EB%A6%BC-%EB%9E%9C%EB%8D%A4-%ED%8A%B8%EB%A6%ACextratreesclassifier/
from sklearn.ensemble import ExtraTreesRegressor
# category encoders 범주형 변수(성별, 종교)를 정수형으로 바꾸어주는 도구
# https://contrib.scikit-learn.org/category_encoders/
import category_encoders as ce
# tensorflow: Google의 ML 라이브러리
import tensorflow as tf
# keras: DL 라이브러리. 텐서플로 위에서 실행 가능.
from tensorflow import keras
# layers? sequential?
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Input, TimeDistributed
from tensorflow.keras.models import Model

print(tf.__version__)
print(keras.__version__)

warnings.filterwarnings(action='ignore')
['NanumGothic']
2.3.0
2.4.0
## 데이터 불러오기
train = pd.read_csv("data/train.csv", encoding = 'euc-kr')
train['DateTime'] = pd.to_datetime(train['DateTime'])
train['date'] = train.DateTime.dt.date

new_train= pd.read_csv("data/2차_train.csv", encoding = 'euc-kr')
new_train['DateTime'] = pd.to_datetime(new_train['DateTime'])
new_train['date'] = new_train.DateTime.dt.date

## 시간 합산
train = pd.DataFrame(train.groupby('date').sum().reset_index())
new_train = pd.DataFrame(new_train.groupby('date').sum().reset_index())

## 1차 train과 2차 train 합치기
train = pd.concat([train, new_train]).reset_index(drop = True)
train_data = train.copy()

## submission 데이터
submission = pd.read_csv('data/submission.csv', encoding = 'euc-kr')
submission.DateTime = pd.to_datetime(submission.DateTime).dt.date
submission = submission[submission.DateTime>date(2020,12,8)].reset_index(drop = True)

# 일별로 데이터를 합친 뒤 트레이닝 데이터를 합침. 1위 코드와 비슷한 과정
```



## 2. 추가 데이터 처리

```python
login_info = pd.read_csv('data/new_login_info.csv', index_col=0)
competition_info = pd.read_csv('data/new_competition_info.csv', index_col=0, encoding='cp949')
user_info = pd.read_csv('data/new_user_info.csv', index_col=0)
submission_info = pd.read_csv('data/new_submission_info.csv', index_col=0).dropna(how = 'all')
## active_made
## id와 c_time만 남김
made_info = user_info[['id','c_time']]

## c_time과 id 모두 같은 곳에서 결측치기에 모두 삭제
made_info_dropna = made_info.dropna().copy()
made_info_dropna['c_time'] = pd.to_datetime(made_info_dropna['c_time'])

# 연월일로 나타내기
made_info_dropna['day'] = made_info_dropna['c_time'].apply(lambda x : x.strftime('%Y-%m-%d') )
made_info_dropna.day = pd.to_datetime(made_info_dropna.day).dt.date
made_info_dropna = made_info_dropna[made_info_dropna.day>=date(2018,9,9)].reset_index(drop = True)

# 아이디 생성한 사용자
active_made = made_info_dropna.groupby(['day'])['id'].nunique().reset_index()
active_made.day = pd.to_datetime(active_made.day).dt.date
active_made.rename(columns = {'id':'active_made'}, inplace = True)


### active_login_user
## user_id와 c_time 변수를 제외한 나머지 모든 변수 제거
## 중간중간 있는 login_id나 user_id의 결측치도 모두 제거
login_info_drop_na = login_info[['user_id','c_time']].dropna().copy()
login_info_drop_na['c_time'] = pd.to_datetime(login_info_drop_na['c_time'])
## 연월일로 나타내기
login_info_drop_na['day'] = login_info_drop_na['c_time'].apply(lambda x : x.strftime('%Y-%m-%d') )
## 실제 로그인한 사용자가 몇 명인가
active_login_user = login_info_drop_na.groupby(['day'])['user_id'].nunique().reset_index()
active_login_user['day'] = pd.to_datetime(active_login_user['day']).dt.date
active_login_user.rename(columns = {'user_id':'active_login_user'}, inplace = True)


### all_login_user
## platform과 browser의 결측치가 많아 변수 제거 -> 유의미한 변수도 아님
## 중간중간 있는 login_id나 user_id의 결측치도 모두 제거
login_info_drop_na2 = login_info[['user_id','c_time']].copy()
## 총 사용자를 확인 하기 위해 결측치 다 채우기
login_info_drop_na2['user_id'].fillna(0, inplace = True)
login_info_drop_na2 = login_info_drop_na2.dropna()
login_info_drop_na2['c_time'] = pd.to_datetime(login_info_drop_na2['c_time'])
## 연월일로 나타내기
login_info_drop_na2['day'] = login_info_drop_na2['c_time'].apply(lambda x : x.strftime('%Y-%m-%d') )
## 해당 일에 로그인 횟수가 총 몇 번인가
all_login_user = login_info_drop_na2.groupby(['day'])['user_id'].count().reset_index()
all_login_user['day'] = pd.to_datetime(all_login_user['day']).dt.date
all_login_user.rename(columns = {'user_id':'all_login_user'}, inplace = True)


## team_id별 user_id별 제출 수 뽑기
sub_info = submission_info.copy()
sub_info_dropna = sub_info[['team_id','user_id','c_time']].dropna().copy()
sub_info_dropna.c_time = pd.to_datetime(sub_info_dropna.c_time)
# 연월로 나타내기
sub_info_dropna['day'] = sub_info_dropna['c_time'].apply(lambda x : x.strftime('%Y-%m-%d') )
sub_info_dropna = sub_info_dropna[(sub_info_dropna['day']>='2018-09-09')]


## 실제 제출한 사용자가 몇 명인가 : user_id
active_sub_user = sub_info_dropna.groupby(['day'])['user_id'].nunique().reset_index()
active_sub_user.day = pd.to_datetime(active_sub_user.day).dt.date
active_sub_user.rename(columns = {'user_id':'active_sub_user'}, inplace = True)

## 팀 아이디로 정리 : 
active_sub_team = sub_info_dropna.groupby(['day'])['team_id'].nunique().reset_index()
active_sub_team.day = pd.to_datetime(active_sub_team.day).dt.date
active_sub_team.rename(columns = {'team_id':'active_sub_team'}, inplace = True)

### team_id별 user_id별 제출 수 뽑기
sub_info = submission_info.copy()
sub_info_all = sub_info[['team_id','user_id','c_time']]
sub_info_all.c_time = pd.to_datetime(sub_info_all.c_time)
sub_info_all[['team_id','user_id']] = sub_info_all[['team_id','user_id']].fillna(0)
sub_info_all['day'] = sub_info_all['c_time'].apply(lambda x : x.strftime('%Y-%m-%d') )
sub_info_all = sub_info_all[(sub_info_all['day']>='2018-09-09')]

## 제출한 총 수
all_sub_user = sub_info_all.groupby(['day'])['user_id'].count().reset_index()
all_sub_user.day = pd.to_datetime(all_sub_user.day).dt.date
all_sub_user.rename(columns = {'user_id':'all_sub_user'}, inplace = True)

## 팀 아이디로 제출한 총 수
all_sub_team = sub_info_all.groupby(['day'])['team_id'].count().reset_index()
all_sub_team.day = pd.to_datetime(all_sub_team.day).dt.date
all_sub_team.rename(columns = {'team_id':'all_sub_team'}, inplace = True)
## 제출수 요일이 가장 적으므로 '제출'을 기준으로 병합
print(active_made.shape)
print(active_login_user.shape)
print(all_login_user.shape)
print(active_sub_user.shape)
print(active_sub_team.shape)
print(all_sub_user.shape)
print(all_sub_team.shape)

add_data_list = [all_login_user, active_sub_user, active_sub_team, all_sub_user, all_sub_team]
add_data = pd.merge(active_made, active_login_user, how = 'inner', on = 'day')
for i in add_data_list:
    add_data = pd.merge(add_data, i, how = 'inner', on = 'day')
    
add_data = add_data.rename(columns = {'day':'date'})
add_data.tail()
(842, 2)
(839, 2)
(839, 2)
(779, 2)
(779, 2)
(779, 2)
(779, 2)
```

|      |       date | active_made | active_login_user | all_login_user | active_sub_user | active_sub_team | all_sub_user | all_sub_team |
| ---: | ---------: | ----------: | ----------------: | -------------: | --------------: | --------------: | -----------: | -----------: |
|  752 | 2021-01-04 |         117 |               387 |            477 |             129 |             118 |          283 |          283 |
|  753 | 2021-01-05 |         101 |               365 |            453 |             131 |             120 |          268 |          268 |
|  754 | 2021-01-06 |         100 |               555 |            639 |             197 |             178 |          430 |          430 |
|  755 | 2021-01-07 |          81 |               420 |            480 |             197 |             174 |          412 |          412 |
|  756 | 2021-01-08 |          82 |               358 |            449 |             205 |             178 |          408 |          408 |



## 3. EDA

### 1) 전체적인 시각화

```python
# train 데이터의 사용자, 세션, 페이지뷰, 신규 방문자 데이터의 기간에 해당하는 추이를 scatter plot으로 나타냄
# go: 그래프를 하나하나 설정하며 그리기 px: 템플릿으로 그래프를 빠르게 제작하기 
# https://data101.oopy.io/plolty-tutorial-guide-in-korean
# .add_trace(): x축과 y축을 각각 날짜와 사용자 수로 설정해준다.
# go.Scatter: go에서 scatter plot과 line plot을 모두 그릴 수 있다.
# showlegend: 범례를 표시할 것인가, 말 것인가

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("User","Session","Page view","New User"), horizontal_spacing=0.08, vertical_spacing=0.08
)
fig.add_trace(go.Scatter(x=train['date'], y=train['사용자'],name="User",showlegend=False, ), row=1, col=1)
fig.add_trace(go.Scatter(x=train['date'], y=train['세션'],name="Session",showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=train['date'], y=train['페이지뷰'],name="Page view",showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=train['date'], y=train['신규방문자'],name="New User",showlegend=False), row=2, col=2)
fig.update_layout(height =700, width =1000, title_text="Scatter Plot for train", template='ggplot2')
fig.show()
```

![image-20220507201611726](2020%20D%20CUP%20Google%20Analytics%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C_2.assets/image-20220507201611726.png)

```python
### 보다 세부적으로 확인
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=list(train.date), y=list(train.사용자),
              name='User'))
fig.add_trace(
    go.Scatter(x=list(train.date), y=list(train.세션),
              name='Session'))
fig.add_trace(
    go.Scatter(x=list(train.date), y=list(train.페이지뷰),
              name='Page veiw'))
fig.add_trace(
    go.Scatter(x=list(train.date), y=list(train.신규방문자),
              name='New User'))

# Set title
fig.update_layout(
    title_text="Overal Time series with Data"
)

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
fig.update_layout(template='ggplot2')
fig.show()
```

Oct 2018Jan 2019Apr 2019Jul 2019Oct 2019Jan 2020Apr 2020Jul 2020Oct 2020050k100k150k

UserSessionPage veiwNew User1m6mYTD1yallOveral Time series with Data













#### 2) 연도별 시각화

```python
### 연도별 Y의 평균
## 연도만 뽑기
train['year'] = train['date'].apply(lambda x : x.strftime('%Y'))

year_train = train.groupby(['year'])[['사용자','세션','신규방문자','페이지뷰']].mean().reset_index()
year_train.head()
```

|      | year |      사용자 |        세션 | 신규방문자 |     페이지뷰 |
| ---: | ---: | ----------: | ----------: | ---------: | -----------: |
|    0 | 2018 |  206.947368 |  204.780702 |  63.000000 |  1618.482456 |
|    1 | 2019 |  506.819178 |  524.487671 | 159.430137 |  6284.134247 |
|    2 | 2020 | 2216.682216 | 2172.446064 | 498.209913 | 55868.854227 |

```python
fig = make_subplots(
    rows=1, cols=4,
    subplot_titles=("train User","train Sess","train Page ","train New"), horizontal_spacing=0.05
)
fig.add_trace(go.Scatter(x=year_train['year'], y=year_train['사용자'],name="User",showlegend=False, ), row=1, col=1)
fig.add_trace(go.Scatter(x=year_train['year'], y=year_train['세션'],name="Session",showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=year_train['year'], y=year_train['페이지뷰'],name="Page view",showlegend=False), row=1, col=3)
fig.add_trace(go.Scatter(x=year_train['year'], y=year_train['신규방문자'],name="New User",showlegend=False), row=1, col=4)
fig.update_layout( title_text="Trend by year", template='ggplot2')
fig.show()
```

20182,018.520192,019.5202050010001500200020182,018.520192,019.5202050010001500200020182,018.520192,019.52020010k20k30k40k50k20182,018.520192,019.52020100200300400500

Trend by yeartrain Usertrain Sesstrain Page train New













#### 3) 월별 시각화

```python
### 연도별 Y의 평균
## 연도만 뽑기
train['month'] = train['date'].apply(lambda x : x.strftime('%Y-%m'))

year_train = train.groupby(['month'])[['사용자','세션','신규방문자','페이지뷰']].mean().reset_index()
year_train.head()
```

|      |   month |     사용자 |       세션 | 신규방문자 |    페이지뷰 |
| ---: | ------: | ---------: | ---------: | ---------: | ----------: |
|    0 | 2018-09 | 226.409091 | 213.409091 |  47.227273 | 1615.590909 |
|    1 | 2018-10 | 149.129032 | 148.774194 |  53.064516 |  909.645161 |
|    2 | 2018-11 | 222.233333 | 226.066667 |  84.033333 | 1704.900000 |
|    3 | 2018-12 | 236.161290 | 234.064516 |  63.774194 | 2245.741935 |
|    4 | 2019-01 | 302.387097 | 300.870968 |  84.032258 | 2434.612903 |

```python
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("train User","train Sess","train Page ","train New"), horizontal_spacing=0.07,vertical_spacing=0.06
)

fig.add_trace(go.Scatter(x=year_train['month'], y=year_train['사용자'],name="User",showlegend=False, ), row=1, col=1)
fig.add_trace(go.Scatter(x=year_train['month'], y=year_train['세션'],name="Session",showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=year_train['month'], y=year_train['페이지뷰'],name="Page view",showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=year_train['month'], y=year_train['신규방문자'],name="New User",showlegend=False), row=2, col=2)
fig.update_layout(height =1000, width =1000, title_text="Trend by month", template='ggplot2')
fig.show()
```

Jan 2019Jul 2019Jan 2020Jul 20200500100015002000250030003500Jan 2019Jul 2019Jan 2020Jul 20200500100015002000250030003500Jan 2019Jul 2019Jan 2020Jul 2020010k20k30k40k50k60k70k80k90kJan 2019Jul 2019Jan 2020Jul 2020100200300400500600700800900

Trend by monthtrain Usertrain Sesstrain Page train New













#### 월의 특징 파악

- 19년 12월부터 20년의 11월까지 특징 그래프.
- 12월 31일, 1월 1일이 떨어진다는 점 보이기 위함.
- 최근의 특징을 가장 많이 가지고 있는 2020년의 데이터에서 월별로 그래프를 그려보았을 때 주말 및 휴일에는 하락하고 평일에는 상승하는 점이 보임
- 특징있는 날을 제외하고서는 한달 이내 상승점들의 높이는 비슷하고 하락점 또한 비슷함.
- 공휴일과 주말이 비슷한 모습을 보임.

```python
train['dayofmonth'] = pd.to_datetime(train['date']).dt.day
train['month'] = pd.to_datetime(train['date']).dt.month
train['year'] = pd.to_datetime(train['date']).dt.year

#### 사용자
#### 가장 최근 월별로 Y 분석을 해본 뒤에 월초와 월말을 어떤식으로 나눌지에 대해서 고민
# 1,2,3,4,5월 분석
train_1_20 = train[(train['month']==1) & (train['year']==2020)]
train_2_20 = train[(train['month']==2) & (train['year']==2020)]
train_3_20 = train[(train['month']==3) & (train['year']==2020)]
train_4_20 = train[(train['month']==4) & (train['year']==2020)]
train_5_20 = train[(train['month']==5) & (train['year']==2020)]
train_6_20= train[(train['month']==6) & (train['year']==2020)]

# 6,7,8,9,10월 분석
train_6_20 = train[(train['month']==6) & (train['year']==2020)]
train_7_20 = train[(train['month']==7) & (train['year']==2020)]
train_8_20 = train[(train['month']==8) & (train['year']==2020)]
train_9_20 = train[(train['month']==9) & (train['year']==2020)]
train_10_20 = train[(train['month']==10) & (train['year']==2020)]
train_11_20 = train[(train['month']==11) & (train['year']==2020)]


# 19년 5~12월 진행
train_4_19 = train[(train['month']==4) & (train['year']==2019)]
train_5_19 = train[(train['month']==5) & (train['year']==2019)]
train_6_19 = train[(train['month']==6) & (train['year']==2019)]
train_7_19 = train[(train['month']==7) & (train['year']==2019)]
train_8_19 = train[(train['month']==8) & (train['year']==2019)]
train_9_19 = train[(train['month']==9) & (train['year']==2019)]
train_10_19 = train[(train['month']==10) & (train['year']==2019)]
train_11_19 = train[(train['month']==11) & (train['year']==2019)]
train_12_19 = train[(train['month']==12) & (train['year']==2019)]

# 18년 10월 11월 12월 19년 1월 2월 4월
train_10_18 = train[(train['month']==10) & (train['year']==2018)]
train_11_18 = train[(train['month']==11) & (train['year']==2018)]
train_12_18 = train[(train['month']==12) & (train['year']==2018)]
train_1_19 = train[(train['month']==1) & (train['year']==2019)]
train_2_19 = train[(train['month']==2) & (train['year']==2019)]
```

- 19년 12월부터 20년의 11월까지 특징 (사용자)

```python
dataset=[train_12_19, train_1_20, train_2_20, train_3_20, train_4_20, train_5_20, 
         train_6_20, train_7_20, train_8_20, train_9_20, train_10_20, train_11_20]

row=6
col=2
num=0

fig = make_subplots(
    rows=row, cols=col,
    subplot_titles=("2019 Dec","2020 Jan","2020 Feb","2020 Mar","2020 Apr","2020 May","2020 Jun", "2020 Jul", "2020 Aug",
                    "2020 Sep","2020 Oct","2020 Nov") ,horizontal_spacing=0.05, vertical_spacing=0.07 )

for i in range(row):
    for j in range(col):
            fig.add_trace(go.Scatter(x=dataset[num]['date'], y=dataset[num]['사용자'],marker=dict(size=9, color='red'),
                                     showlegend=False,name='User'),row=i+1,col=j+1)
            num+=1
            
fig.update_layout(height =1500, width =1000, title_text="User Trend by month (2019 Dec ~ 2020 Oct)", template='ggplot2')
fig.show()
```

Dec 12019Dec 8Dec 15Dec 22Dec 290500100015002000Jan 52020Jan 12Jan 19Jan 260100020003000Feb 22020Feb 9Feb 16Feb 23100015002000Mar 12020Mar 8Mar 15Mar 22Mar 29150020002500Apr 52020Apr 12Apr 19Apr 2615002000May 32020May 10May 17May 24May 3110001500200025003000Jun 72020Jun 14Jun 21Jun 281500200025003000Jul 52020Jul 12Jul 19Jul 262000300040005000Aug 22020Aug 9Aug 16Aug 23Aug 3020003000Sep 62020Sep 13Sep 20Sep 27200030004000Oct 42020Oct 11Oct 18Oct 25200030004000Nov 12020Nov 8Nov 15Nov 22Nov 292000300040005000

User Trend by month (2019 Dec ~ 2020 Oct)2019 Dec2020 Jan2020 Feb2020 Mar2020 Apr2020 May2020 Jun2020 Jul2020 Aug2020 Sep2020 Oct2020 Nov













- 19년 12월부터 20년의 11월까지 특징 (세션)

```python
dataset=[train_12_19, train_1_20, train_2_20, train_3_20, train_4_20, train_5_20, 
         train_6_20, train_7_20, train_8_20, train_9_20, train_10_20, train_11_20]

row=6
col=2
num=0

fig = make_subplots(
    rows=row, cols=col,
    subplot_titles=("2019 Dec","2020 Jan","2020 Feb","2020 Mar","2020 Apr","2020 May","2020 Jun", "2020 Jul", "2020 Aug",
                    "2020 Sep","2020 Oct","2020 Nov") ,horizontal_spacing=0.05, vertical_spacing=0.07 )

for i in range(row):
    for j in range(col):
            fig.add_trace(go.Scatter(x=dataset[num]['date'], y=dataset[num]['세션'],marker=dict(size=9, color='Blue'),
                                     showlegend=False, name='Session'),row=i+1,col=j+1)
            num+=1
            
fig.update_layout(height =1500, width =1000, title_text="Session Trend by month (2019 Dec ~ 2020 Oct)", template='ggplot2')
fig.show()
```

Dec 12019Dec 8Dec 15Dec 22Dec 290500100015002000Jan 52020Jan 12Jan 19Jan 260100020003000Feb 22020Feb 9Feb 16Feb 23100015002000Mar 12020Mar 8Mar 15Mar 22Mar 291000150020002500Apr 52020Apr 12Apr 19Apr 2615002000May 32020May 10May 17May 24May 3110001500200025003000Jun 72020Jun 14Jun 21Jun 281500200025003000Jul 52020Jul 12Jul 19Jul 262000300040005000Aug 22020Aug 9Aug 16Aug 23Aug 3020003000Sep 62020Sep 13Sep 20Sep 27200030004000Oct 42020Oct 11Oct 18Oct 25200030004000Nov 12020Nov 8Nov 15Nov 22Nov 292000300040005000

Session Trend by month (2019 Dec ~ 2020 Oct)2019 Dec2020 Jan2020 Feb2020 Mar2020 Apr2020 May2020 Jun2020 Jul2020 Aug2020 Sep2020 Oct2020 Nov













- 19년 12월부터 20년의 11월까지 특징 (신규방문자)

```python
dataset=[train_12_19, train_1_20, train_2_20, train_3_20, train_4_20, train_5_20, 
         train_6_20, train_7_20, train_8_20, train_9_20, train_10_20, train_11_20]

row=6
col=2
num=0

fig = make_subplots(
    rows=row, cols=col,
    subplot_titles=("2019 Dec","2020 Jan","2020 Feb","2020 Mar","2020 Apr","2020 May","2020 Jun", "2020 Jul", "2020 Aug",
                    "2020 Sep","2020 Oct","2020 Nov") ,horizontal_spacing=0.05, vertical_spacing=0.07 )
for i in range(row):
    for j in range(col):
            fig.add_trace(go.Scatter(x=dataset[num]['date'], y=dataset[num]['신규방문자'],marker=dict(size=9, color='Purple'),
                                     showlegend=False, name='New User'),row=i+1,col=j+1)
            num+=1
            
fig.update_layout(height =1500, width =1000, title_text="New User Trend by month (2019 Dec ~ 2020 Oct)", template='ggplot2')
fig.show()
```

Dec 12019Dec 8Dec 15Dec 22Dec 29200400600Jan 52020Jan 12Jan 19Jan 260200400600Feb 22020Feb 9Feb 16Feb 23200300400Mar 12020Mar 8Mar 15Mar 22Mar 29200400600Apr 52020Apr 12Apr 19Apr 26200400600May 32020May 10May 17May 24May 315001000Jun 72020Jun 14Jun 21Jun 28400600800Jul 52020Jul 12Jul 19Jul 2650010001500Aug 22020Aug 9Aug 16Aug 23Aug 304006008001000Sep 62020Sep 13Sep 20Sep 2740060080010001200Oct 42020Oct 11Oct 18Oct 252004006008001000Nov 12020Nov 8Nov 15Nov 22Nov 2950010001500

New User Trend by month (2019 Dec ~ 2020 Oct)2019 Dec2020 Jan2020 Feb2020 Mar2020 Apr2020 May2020 Jun2020 Jul2020 Aug2020 Sep2020 Oct2020 Nov













- 19년 12월부터 20년의 11월까지 특징 (페이지뷰)

```python
dataset=[train_12_19, train_1_20, train_2_20, train_3_20, train_4_20, train_5_20, 
         train_6_20, train_7_20, train_8_20, train_9_20, train_10_20, train_11_20]

row=6
col=2
num=0

fig = make_subplots(
    rows=row, cols=col,
    subplot_titles=("2019 Dec","2020 Jan","2020 Feb","2020 Mar","2020 Apr","2020 May","2020 Jun", "2020 Jul", "2020 Aug",
                    "2020 Sep","2020 Oct","2020 Nov") ,horizontal_spacing=0.05, vertical_spacing=0.07 )
for i in range(row):
    for j in range(col):
            fig.add_trace(go.Scatter(x=dataset[num]['date'], y=dataset[num]['페이지뷰'],marker=dict(size=9, color='Green'),
                                     showlegend=False,name='Page view'),row=i+1,col=j+1)
            num+=1
            
fig.update_layout(height =1500, width =1000, title_text="Page view Trend by month (2019 Dec ~ 2020 Oct)", template='ggplot2')
fig.show()
```

Dec 12019Dec 8Dec 15Dec 22Dec 29010k20k30kJan 52020Jan 12Jan 19Jan 26010k20k30kFeb 22020Feb 9Feb 16Feb 2320k30k40k50kMar 12020Mar 8Mar 15Mar 22Mar 2940k60k80k100kApr 52020Apr 12Apr 19Apr 2630k40k50k60k70kMay 32020May 10May 17May 24May 3140k60k80kJun 72020Jun 14Jun 21Jun 2840k60k80k100kJul 52020Jul 12Jul 19Jul 2650k100k150kAug 22020Aug 9Aug 16Aug 23Aug 3040k60k80kSep 62020Sep 13Sep 20Sep 2740k60k80k100kOct 42020Oct 11Oct 18Oct 2540k60k80k100kNov 12020Nov 8Nov 15Nov 22Nov 2950k100k150k

Page view Trend by month (2019 Dec ~ 2020 Oct)2019 Dec2020 Jan2020 Feb2020 Mar2020 Apr2020 May2020 Jun2020 Jul2020 Aug2020 Sep2020 Oct2020 Nov













#### 연휴 기간

- 연휴 기간에도 휴일과 같이 하락한 모습을 보임
- 20년 5월의 연휴 확인
- 19년 12월 31일 연말 모습 확인
- 18년 12/31, 19년 12/31, 추가 데이터의 제출 수, 로그인 수, 아이디 생성 수 등을 시각화로 보이기

```python
## 18년과 19년 12/31 ~ 1/1
train_later_18 = train[(train['date']>=date(2018, 12, 23))<=date(2019, 1, 5))]
train_later_19 = train[(train['date']>=date(2019, 12, 23))<=date(2020, 1, 5))]

## 추가데이터 12/31 ~ 1/1
add_later18 = add_data[(add_data['date']>=date(2018, 12, 20))<=date(2019, 1, 5))]
add_later19 = add_data[(add_data['date']>=date(2019, 12, 20))<=date(2020, 1, 5))]
## train 데이터의 연말 연초(사용자, 세션, 신규방문자)

fig = make_subplots(
    subplot_titles=("2018.12.23 ~ 2019.1.5", "2019.12.13 ~ 2020.1.5"),
    horizontal_spacing=0.07,
    rows=1, cols=2 )

fig.add_trace(go.Scatter(x=train_later_18['date'], y=train_later_18['사용자'],name="User",
                         marker=dict(size=9, color='Red')), row=1, col=1)
fig.add_trace(go.Scatter(x=train_later_18['date'], y=train_later_18['세션'],name="Session",
                         marker=dict(size=9, color='Green')), row=1, col=1)
fig.add_trace(go.Scatter(x=train_later_18['date'], y=train_later_18['신규방문자'],name="New User",
                         marker=dict(size=9, color='Blue')), row=1, col=1)

fig.add_trace(go.Scatter(x=train_later_19['date'], y=train_later_19['사용자'],name="User",
                         marker=dict(size=9, color='Red'),showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=train_later_19['date'], y=train_later_19['세션'],name="Session",
                         marker=dict(size=9, color='Green'),showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=train_later_19['date'], y=train_later_19['신규방문자'],name="New User",
                         marker=dict(size=9, color='Blue'),showlegend=False), row=1, col=2)
fig.update_layout(title_text="Trend by End to Beginning Year (User, Session, New User)", template='ggplot2')
fig.show()



## train 연말 연초 (페이지뷰)
fig_2 = make_subplots(
    subplot_titles=("2018.12.23 ~ 2019.1.5", "2019.12.13 ~ 2020.1.5"),
    horizontal_spacing=0.07,
    rows=1, cols=2,
)

fig_2.add_trace(go.Scatter(x=train_later_18['date'], y=train_later_18['페이지뷰'],name="Page view",
                         marker=dict(size=7, color='Purple')), row=1, col=1)
fig_2.add_trace(go.Scatter(x=train_later_19['date'], y=train_later_19['페이지뷰'],name="Page view",
                         marker=dict(size=7, color='Purple'),showlegend=False), row=1, col=2)
fig_2.update_layout(title_text="Trend by End to Beginning Year (Page view)", template='ggplot2')
fig_2.show()
```

Dec 232018Dec 26Dec 29Jan 12019Jan 40100200300400500Dec 242019Dec 27Dec 30Jan 22020Jan 50500100015002000

UserSessionNew UserTrend by End to Beginning Year (User, Session, New User)2018.12.23 ~ 2019.1.52019.12.13 ~ 2020.1.5













Dec 232018Dec 26Dec 29Jan 12019Jan 4100020003000400050006000Dec 242019Dec 27Dec 30Jan 22020Jan 505k10k15k20k

Page viewTrend by End to Beginning Year (Page view)2018.12.23 ~ 2019.1.52019.12.13 ~ 2020.1.5













- 신규 아이디 생성자(active_made)와 실제 팀, 사용자 제출 수는 18년 연말 모두 하락세
- 전체적으로 수치가 낮으나 크리스마스와 1월 1일이 가장 낮은 것을 알 수 있음.
- 평일에 비해 크리스마스, 주말, 12월 31일은 모두 낮은 모습을 보임

```python
## train 데이터의 연말 연초(사용자, 세션, 신규방문자)
fig = make_subplots(
    subplot_titles=("2018.12.23 ~ 2019.1.5", "2019.12.13 ~ 2020.1.5"),
    horizontal_spacing=0.07,
    rows=1, cols=2 )

fig.add_trace(go.Scatter(x=add_later18['date'], y=add_later18['active_made'],name="active_made",
                         marker=dict(size=9, color='Red')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_later18['date'], y=add_later18['all_login_user'],name="all_login_user",
                         marker=dict(size=9, color='Purple')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_later18['date'], y=add_later18['active_sub_user'],name="active_sub_user",
                         marker=dict(size=9, color='Orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_later18['date'], y=add_later18['active_sub_team'],name="active_sub_team",
                         marker=dict(size=9, color='Black')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_later18['date'], y=add_later18['all_sub_user'],name="all_sub_user",
                         marker=dict(size=9, color='Green')), row=1, col=1)


fig.add_trace(go.Scatter(x=add_later19['date'], y=add_later19['active_made'],name="active_made",
                         marker=dict(size=7, color='Red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_later19['date'], y=add_later19['all_login_user'],name="all_login_user",
                         marker=dict(size=7, color='Purple'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_later19['date'], y=add_later19['active_sub_user'],name="active_sub_user",
                         marker=dict(size=7, color='Orange'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_later19['date'], y=add_later19['active_sub_team'],name="active_sub_team",
                         marker=dict(size=7, color='Black'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_later19['date'], y=add_later19['all_sub_user'],name="all_sub_user",
                         marker=dict(size=7, color='Green'), showlegend=False), row=1, col=2)


fig.update_layout(title_text="Trend by End to Beginning Year (user info)", template='ggplot2')
fig.show()
```

Dec 202018Dec 23Dec 26Dec 29Jan 12019Jan 4020406080100120140160Dec 212019Dec 24Dec 27Dec 30Jan 22020Jan 50100200300400500

active_madeall_login_useractive_sub_useractive_sub_teamall_sub_userTrend by End to Beginning Year (user info)2018.12.23 ~ 2019.1.52019.12.13 ~ 2020.1.5













- Y 모두 20년의 경우 석가탄신일을 기점으로 하락하면서 연휴 기간 내내 하락세
- 로그인 수, 제출 수 등 연휴 기간 동안 하락세
- 공휴일 휴일이 아니더라도 긴 연휴에 포함되는 날은 공휴일 및 주말과 같은 패턴을 보임

```python
## 19년과 20년 5월 연휴
train_5_19 = train[(train['date']>=date(2019, 4, 25))<=date(2019, 5, 10))]
train_5_20 = train[(train['date']>=date(2020, 4, 25))<=date(2020, 5, 10))]

## 추가데이터 5월 연휴
add_5_18 = add_data[(add_data['date']>=date(2019, 4, 25))<=date(2019, 5, 10))]
add_5_19 = add_data[(add_data['date']>=date(2020, 4, 25))<=date(2020, 5, 10))]
fig = make_subplots(
    subplot_titles=("2019.4.25 ~ 2019.5.10", "2020.4.25 ~ 2020.5.10"),
    horizontal_spacing=0.07,
    rows=1, cols=2 )

fig.add_trace(go.Scatter(x=train_5_19['date'], y=train_5_19['사용자'],name="User",
                         marker=dict(size=9, color='Red')), row=1, col=1)
fig.add_trace(go.Scatter(x=train_5_19['date'], y=train_5_19['세션'],name="Session",
                         marker=dict(size=9, color='Green')), row=1, col=1)
fig.add_trace(go.Scatter(x=train_5_19['date'], y=train_5_19['신규방문자'],name="New User",
                         marker=dict(size=9, color='Blue')), row=1, col=1)

fig.add_trace(go.Scatter(x=train_5_20['date'], y=train_5_20['사용자'],name="User",
                         marker=dict(size=9, color='Red'),showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=train_5_20['date'], y=train_5_20['세션'],name="Session",
                         marker=dict(size=9, color='Green'),showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=train_5_20['date'], y=train_5_20['신규방문자'],name="New User",
                         marker=dict(size=9, color='Blue'),showlegend=False), row=1, col=2)
fig.update_layout(title_text="Trend affected by Holiday (User, Session, New User)", template='ggplot2')
fig.show()



##  (페이지뷰)
fig_2 = make_subplots(
    subplot_titles=("2019.4.25 ~ 2019.5.10", "2020.4.25 ~ 2020.5.10"),
    horizontal_spacing=0.07,
    rows=1, cols=2,
)
fig_2.add_trace(go.Scatter(x=train_5_19['date'], y=train_5_19['페이지뷰'],name="Page view",
                         marker=dict(size=7, color='Purple')), row=1, col=1)
fig_2.add_trace(go.Scatter(x=train_5_20['date'], y=train_5_20['페이지뷰'],name="Page view",
                         marker=dict(size=7, color='Purple'),showlegend=False), row=1, col=2)
fig_2.update_layout(title_text="Trend affected by Holiday (Page view)", template='ggplot2')
fig_2.show()
```

Apr 252019Apr 28May 1May 4May 7May 1002004006008001000120014001600Apr 252020Apr 28May 1May 4May 7May 10500100015002000

UserSessionNew UserTrend affected by Holiday (User, Session, New User)2019.4.25 ~ 2019.5.102020.4.25 ~ 2020.5.10













Apr 252019Apr 28May 1May 4May 7May 102k4k6k8k10k12k14k16kApr 252020Apr 28May 1May 4May 7May 1030k35k40k45k50k55k

Page viewTrend affected by Holiday (Page view)2019.4.25 ~ 2019.5.102020.4.25 ~ 2020.5.10













```python
## train 데이터의 연말 연초(사용자, 세션, 신규방문자)
fig = make_subplots(
    subplot_titles=("2019.4.25 ~ 2019.5.10", "2020.4.25 ~ 2020.5.10"),
    horizontal_spacing=0.07,
    rows=1, cols=2 )

fig.add_trace(go.Scatter(x=add_5_18['date'], y=add_5_18['active_made'],name="active_made",
                         marker=dict(size=9, color='Red')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_5_18['date'], y=add_5_18['all_login_user'],name="all_login_user",
                         marker=dict(size=9, color='Purple')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_5_18['date'], y=add_5_18['active_sub_user'],name="active_sub_user",
                         marker=dict(size=9, color='Orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_5_18['date'], y=add_5_18['active_sub_team'],name="active_sub_team",
                         marker=dict(size=9, color='Black')), row=1, col=1)
fig.add_trace(go.Scatter(x=add_5_18['date'], y=add_5_18['all_sub_user'],name="all_sub_user",
                         marker=dict(size=9, color='Green')), row=1, col=1)


fig.add_trace(go.Scatter(x=add_5_19['date'], y=add_5_19['active_made'],name="active_made",
                         marker=dict(size=9, color='Red'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_5_19['date'], y=add_5_19['all_login_user'],name="all_login_user",
                         marker=dict(size=9, color='Purple'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_5_19['date'], y=add_5_19['active_sub_user'],name="active_sub_user",
                         marker=dict(size=9, color='Orange'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_5_19['date'], y=add_5_19['active_sub_team'],name="active_sub_team",
                         marker=dict(size=9, color='Black'), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=add_5_19['date'], y=add_5_19['all_sub_user'],name="all_sub_user",
                         marker=dict(size=9, color='Green'), showlegend=False), row=1, col=2)


fig.update_layout(title_text="Trend by End to Beginning Year (user info)", template='ggplot2')
fig.show()
```

May 22019May 4May 6May 8May 10050100150200Apr 252020Apr 28May 1May 4May 7May 10050100150200250300

active_madeall_login_useractive_sub_useractive_sub_teamall_sub_userTrend by End to Beginning Year (user info)2019.4.25 ~ 2019.5.102020.4.25 ~ 2020.5.10













#### Y별 가장 큰 값들이 존재하는 날의 특징 파악

- 휴일과 평일의 오르내림과 달리 유난히 높은 값을 갖는 날이 확인됨(7/31)
- 2020년부터 본격적인 상승세를 타는 모습에서 어떠한 패턴을 가지고 있지 않기에 해당 날만이 가진 특수성을 파악해야 한다고 생각
- 월간 데이콘과 같은 대회가 아닌 '코로나 대회'등과 같은 일정한 패턴이 없고 강한 특수성을 가진 대회의 정보는 반영하기 어렵다고 판단
- 추가 데이터를 통해 확인할 수 있는 '로그인 수','제출 수','아이디 생성 수','참가자 수','대회 마감일'등의 정보를 종합하여 폭발적으로 증가하는 날을 잡아낼 수 있는지 확인

```python
## 대회 시작일과 마감일 뽑기
top_competition = competition_info[['period_start','period_end','name','participants','merge_deadline']].dropna()
for i in ['period_start','period_end','merge_deadline']:
    top_competition[i] = pd.to_datetime(top_competition[i]).dt.date

top_user = list(train.sort_values(by = '사용자', ascending = False)['date'].head())
top_sess = list(train.sort_values(by = '세션', ascending = False)['date'].head())
top_new = list(train.sort_values(by = '신규방문자', ascending = False)['date'].head())
top_page = list(train.sort_values(by = '페이지뷰', ascending = False)['date'].head())

top_list = list(set(top_user)|set(top_sess)|set(top_new)|set(top_page))

df=[]
for i in ['period_end','merge_deadline']:
    print(top_competition[top_competition[i].isin(top_list)].groupby(i)['participants'].sum())
period_end
2020-07-31    3572.0
2020-11-16    1696.0
Name: participants, dtype: float64
merge_deadline
2020-11-02    1298.0
2020-11-25     682.0
Name: participants, dtype: float64
```

- 11월 25일을 제외하고 각 Y에서 높은 값을 갖는 날들은 팅 병합 마감하거나 끝나는 대회가 1000명 이상인 것으로 보인다.
- test기간 중 참가자가 1000명이 넘는 대회가 끝나거나 팀 병합이 마감하는지 확인
- 만약 존재한다면 위의 가정을 바탕으로 알고리즘의 정확도를 대략적으로 검증 가능
- test 기간 중에서는 1000명 이상의 대회가 끝나는 12월 31일과 1000명 이상이 참가하는 대회의 시작일인 1월 6일이 포함되어 있음

```python
## test 기간 확인
start = top_competition[['period_start','participants']]
start_sum = start.groupby('period_start').sum().reset_index()
start_sum = start_sum[start_sum.participants>1000].reset_index(drop = True)

end = top_competition[['period_end','participants']]
end_sum = end.groupby('period_end').sum().reset_index()
end_sum = end_sum[end_sum.participants>1000].reset_index(drop = True)

deadline = top_competition[['merge_deadline','participants']]
deadline_sum = deadline.groupby('merge_deadline').sum().reset_index()
deadline_sum = deadline_sum[deadline_sum.participants>1000].reset_index(drop = True)

## train 데이터의 연말 연초(사용자, 세션, 신규방문자)
plt.figure(figsize = (15,5))
plt.plot(start_sum.period_start, start_sum.participants, linewidth=3, label = 'end_sum')
plt.plot(end_sum.period_end, end_sum.participants, linewidth=3, label = 'end_sum')
plt.plot(deadline_sum.merge_deadline, deadline_sum.participants, linewidth=3, label = 'deadline_sum')
plt.axvline(x=date(2020,12,9), color='black', linewidth=5, linestyle=':')
plt.legend(loc='upper left')
plt.title('test>1000')
plt.show()
```

![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA3IAAAE+CAYAAADI0iLfAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdeXyU1d3//9eZmeyENWxJCKsKslkUFEQQxaW2VutPR9x627uKttVWkbqhgrjU2moXaxftcvuttTiirW21VhEFFXADF0CQPRtbQiD7Mpnz++OaTCaEhIQsM5O8n48HDzLnuuaakxyuMO/5XNc5xlqLiIiIiIiIxA5XpDsgIiIiIiIiraMgJyIiIiIiEmMU5ERERERERGKMgpyIiIiIiEiMUZATERERERGJMQpyIiLSLRlj0iPdBxERkWOlICciIt3VS8aYU462kzFmoDFmlTFmSRPbzzbGfGCMWW+MWWeM+cYR9vmBMWZD8M9bxpjRh21PMsb8xhizyRiz2RjzjDEm9di/NRER6eoU5EREJKoZY+42xnja4Tg3GWP6hjXdhhPmpjbznInAW0Au0KgPxpjjgD8D11hrxwGXAL8yxpwUts/VwGXAVGvtWOBR4FVjTErYoZ4AqoATgdFATvC4IiIiR2S0ILiIiEQzY4wFUq21pW08zk7g69ba9WFtk4EXcYLYiiM85wJgFzA5+NxLD9v+E+CAtfYnYW03ASdaa78XfPw+8ANr7fth+ywFfNZanzGmF/AlMNRaWxnc7gGygUnW2j1t+b5FRKRrUkVORESikjHme8aYutD1gTHm+WD7DcHLD78MXvJ4Sthzrgtevvi5MeYTY8wEY8xFweOkA383xrxTt7+19kPgG8D/M8ace3gfrLWvWms3NNPN84A3D2tbDpwb7E8fnArbB03tA8wCPqwLccHX9QPvArObeW0REenG2nypioiISEew1v4G+E2wIjfFWltqjLkQuAI4zVpbZIw5HefyyBOA/sAiYJy19qAxxgSP8xnwcrAi983wilxw+yfBytsrxpibrLX/bkU3M3AqduF24YRGgn/n2saXv+wCvt7MMQ4/joiISAOqyImISCy5A7jdWlsEYK19D9iIU9VyASb4NzaoJQcNVt3OBX5njBnRiv70wbm3LVwFkGSMSWhie90+fZo5xuH7iIiINKCKnIiIxJKxwB+MMYGwtl5Ab2vtTmPMgziXYT4D/Mpae6gVxz4JJ1CVtOI5lUDCYW1JgAWqm9het09deGtun7JW9EVERLoRVeRERCTWfN1ae1LYn+HW2ucArLW/BSbhfFC5yRhzfEsOaIz5FvAgcLa1dn8r+pIDDD2sLQvIC1YDc4AhdZd5HrZPTjPHOHwfERGRBhTkREQk2oVX37YApzW3s7W22Fq7EHgG+G7Yptoj7W+MmQvcCcyy1u5sZd/eAc4+rO1snIlKsNbuBfYAU5raB3gPmGqMSQzrkweYEbaPiIhIAwpyIiIS7QqBuvvWngB+HL6gdt09bcFFtXsEv04AxuOs/3ak49Q99ybg+8CZ1tq8Y+jbE8DNdZU/Y8ww4EfAL8P2eQz4Wd0C38aY83HCaF0VMRd4HXjYGOMKVu8WAmuttZuPoU8iItIN6B45ERGJdo8A/zLG7MGZsfIngM8YEw/UAKuBuTj3z/3TGFN3X9nrwK/CjvMYzmLd9wDzcMLU5TiVuANH6UN18E8D1trPjTE3AM8bY+Jw7o37kbV2Tdg+vw8uQ/BBcAbOA8DXrLXh9+LNxQl/X+BM2PIxcM1R+iQiIt2YFgQXEZFuyRjzV+B7rZwQRUREJCooyImISLdkjDEtXZ5AREQk2ijIiYiIiIiIxBhNdiIiIiIiIhJjFORERERERERiTDTPWqlrPkVEREREpLszR2qM5iBHfn5+ux8zLS2NgoKCdj+utJ3GJvpoTKKXxiZ6aWyin8YoemlsopvGp/Olp6c3uS2qg5yIiIiISHfw2GOPNWq77bbbItATiRUKciIiIiIiEfb44483alOQk+ZoshMREREREZEYE1MVOWstlZWVBAIBjDniPX9HtXfvXqqqqtq5Z92PtRaXy0ViYuIxj4WIiIiIiBybmApylZWVxMXF4fEce7c9Hg9ut7sde9V9+f1+KisrSUpKinRXRERERES6lZgKcoFAoE0hTtqXx+NRdVNERESkHcybNy/SXZAYE1OpSJfwRR+NiYiIiEjbaWITaS1NdiIiIiIiIhJjFOQ60dVXXx3pLoiIiLRIbcBSVl0b6W6IiEgTYurSylhXU1MT6S6IiIgc1aFKP3e8vov9ZTVcNaE/l4ztF+kuiYjIYWI2yNVe/41je14L9nE//c9mtz/yyCOUlJRQVlbGlVdeyZIlS0hLS6O8vJy9e/dy4403cvLJJ5OTk8OiRYvIyMggLi6u2YlBCgoKuP/+++nbty+ZmZlcf/31XH311Tz77LMA5OXl8ctf/pJHH32UefPmkZ6eTr9+/di0aROTJk2ipKSEnTt3MmfOHMaNG9eaH4mIiEgDL208wO4S58PHZz7Zjz9g8Y5Pi3CvREQkXMwGuUhZvnw5KSkp3Hnnnfj9fq699lrS0tKYOXMmp59+Ovv27WPBggU8/fTTPPnkk8yfP58xY8aQn5/Pa6+91uRxN27cSHp6OnfddVeoLbyCFwgECAQCoa/PO+88xo8fz3PPPce2bdu4++67yc/P5xe/+AWPPvpox/0ARESkSyuq8PPql0UN2v76WQEWuFxhTkQkaijItdKmTZvYuHEjDz/8MADx8fEADB8+HIABAwZQXFwMQG5uLqNHjwYgPT2dtLSm/wOcMWMGpaWl3HHHHVx44YVMnz69wfba2oa1xH79nMtcEhISGDlyJACJiYlUVFS09VsUEZFu7O8bC6mutQAYwAbbn/usAGthzgSFORGRaBCzQe5olz82xePx4Pf7j/l1hw8fTnx8PNddd12o7ZZbbjniviNGjGDz5s2MHj2anTt3UlBQ0OyxL7jgAs4//3wuu+wypk+fTnJyMgUFBaSlpbFu3bpj7rOIiEhLFFX4+c+Wg6HHt52ezrJtB/lkTzkAf/u8AIvlign9I9VFkS5r1apVjdqmTZsWgZ5IrGhRkPN6vU8G900FvvT5fIu8Xu8yYGvYbnf6fL6DXq93IvAwUAqUA3N9Pl9NU+3t+L10ivPOO4+FCxcyb9484uPjmTJlCm63G7fbHdonLi4OgJtuuon777+ffv36kZKSQlZWVpPHXbNmDUuWLMHj8XDqqacCcN111zF//nxGjRpFampq6DXCX8/tdmNt8JNTYxr0Q0REpDXCq3Ej+iQwfWgqUzJ78PDKPD7ZXQbAks8LscAV49O0lqhIO7rssssateXl5UWgJxIrTF0IaCmv1/sMTiB70ufzzT7C9leAa3w+3wGv13sdYHw+39NNtTfzUjY/P79BQ3l5OcnJya3q7+HaWpGThtpjTOqkpaUdtWopnUtjEr00NtErVsemqMLP3Je3hYLc3TMzODUzFYDq2gA/XpHH2mCYA/CO68eVE2IzzMXqGHUH3XlsMjIyGrVFW5DrzuMTKenp6eBc6d5Iqy6t9Hq9vYA0YC9Q4vV67wOygPd8Pt+fvV5vIuD3+XwHgk/5B/Arr9f7lyO1A80FuS5px44d+Hy+Bm3GGH74wx+SkJAQoV6JiEh391JYNW5k3wSmZPQIbYt3u7hrZgaPrMzj43wnzPnWFxKwcPXE2AxzIiKxrqWXVo4C7gemADf7fL6DwDeD2wzwpNfr3QF8CRwMe+oBoG/wz5HaD3+ducBcAJ/P12hykL179+LxtP22vvY4xrE67rjjWLBgQcRev70lJCQ0O4lLa3g8nnY7lrQPjUn00thEr1gcm8Kyal7b8mXo8dzTR9C/f+O14376zTQWvPIFq3c6s1ou3VBIUlISN0wbGlNhLhbHqLvQ2DQUbT8LjU90aVGi8fl8W4GrvF6vB/ib1+v9xOfz7Qlus8HLJicCq4E+YU/tixPaCptoP/x1ngKeCj60h5duq6qq2nwPmC6tbF9VVVXtVmJXuT76aEyil8YmesXi2Pzx471U1zpL3Izsm8AJqYEmv4fbTuvPI9XVfBSszP3lo1zKysv51kn9YybMxeIYdRfdeWymTp3aqC3afhbdeXwiJXhp5RG5WnMgn8/nB9xA/GGbZgAf+Xy+KiDe6/XWVdsuBlY01d6a1xYREZH2V1Th57WwmSrnHGUSkzi3iztnZDA57NLLlzYe4Jl1+2ntffciUm/p0qWN/og056gVOa/XOwmYhzPbZArwos/ny/Z6vY8HHycC7/t8vveCT7kd+KPX6y0BqoCbjtIuIiIiEdLw3rjEBgGtKXFuF3eckc6j7+bzQW4pAH//4gAWuPYrsVOZExGJZa2etbITadbKGKBZK7s2jUn00thEr1gamwMVfm4Im6nynpmZTM48epCrU1Nr+em7ebwfDHMAF43uw7cnDYjqMBdLY9TdaGyim8an8zU3a2WrLq2Utrn66qsj3QUREZGQw6txp2SktOr5cW7Dj6ZncNqQ+vD38qYi/rh2ny6zFBHpYApynaimJubWPxcRkS7qQIWf/4bdG3esC3zXhbmpYWHuX5uK+OPHCnMiIh0pcvPwt9FFf93UYcd++arRzW5/5JFHKCkpoaysjCuvvJIlS5aQlpZGeXk5e/fu5cYbb+Tkk08mJyeHRYsWkZGRQVxcHFVVVU0es6CggPvvv5++ffuSmZnJ9ddfz9VXX82zzz4LOAtC/vKXv+TRRx9l3rx5pKen069fPzZt2sSkSZMoKSlh586dzJkzh3HjxrXrz0NERLqe8GrcqGOoxoXzuAzzp2fws3fzWZ1TAsC/NhcRAK4/ObovsxQRiVUxG+QiZfny5aSkpHDnnXfi9/u59tprSUtLY+bMmZx++uns27ePBQsW8PTTT/Pkk08yf/58xowZQ35+Pq+99lqTx924cSPp6encddddobbwCl4gECAQCIS+Pu+88xg/fjzPPfcc27Zt4+677yY/P59f/OIXPProox33AxARkZjXqBo3oe2LejthLp3H38vnvWwnzL2yuQhrLXNPGagwJyLSzhTkWmnTpk1s3LiRhx9+GID4eGclhuHDhwMwYMAAiouLAcjNzWX0aKe6l56e3uwCijNmzKC0tJQ77riDCy+8kOnTpzfYXltb2+Bxv37OQq0JCQmMHDkSgMTERCoqKtr6LYqISBf30oaG1biT04+9GhfO4zLcdno6UB/mXv3yINbCDZMV5kRE2lPMBrmjXf7YlLbOWjl8+HDi4+O57rrrQm233HLLEfcdMWIEmzdvZvTo0ezcufOos/xccMEFnH/++Vx22WVMnz6d5ORkCgoKSEtLY926dcfcZxERkToHKvz8d2v7VuPCuYNhzmXyeWeXE+b+E6z+zZ08EJfCnMgRZWRkNGrLy8uLQE8kVsRskIuU8847j4ULFzJv3jzi4+OZMmUKbrcbt9sd2icuLg6Am266ifvvv59+/fqRkpJCVlZWk8dds2YNS5YswePxcOqppwJw3XXXMX/+fEaNGkVqamroNcJfz+12h24mN8Y06IeIiMjhOqoaF87tMtw6LR3Dblbucq5S+c+WgwQs3DhFYU5EpD1oHTlpE60j17VpTKKXxiZ6RfPYHL5u3L1nZnJKCxYAP1a1AcsvVu9m5c7iUNu5o3rx3SmDIhrmonmMurvuPDaxUJHrzuMTKc2tI6eKXCfbsWMHPp+vQZsxhh/+8IckJCREqFciItIdvBhWjTuuX8dU48K5XYZbpg7GBbwdDHOvbz2EtfC9UyMb5kREYp2CXCcbPnw4d9xxR6S7ISIi3UxheU2DmSrnHOO6ca3ldhl+MHUwGHh7hxPm3th2CAt8X2FOROSYKciJiIh0Ay9tPEBNoPOqceHcLsMPThuMy8Dy7U6YW7bNqczddJrCnAhE32WUEv1cke6AiIiIdKxIVePCuV2Gm04dzFkjeoXa3tx+iCfW7KE2ELX364uIRC0FORERkS4uktW4cG6X4ebTBjF7ZH2YW779EE+s2a0wJyLSSgpyIiIiXdjh1bgrIlCNC+cyhu+f2jDMvbWjmF8pzImItIqCXBs98cQTvP/++21+fl5eHrfffjsAt99+OxUVFe3VRRER6cZePKwaNylC1bhwdWHunLAw9/aOYn65WmFORKSlNNlJG9XW1lJbW9vm5wcCAQKBAACPPvpoe3VPRES6scLyGl6PompcOJcxoSUI/rvV6eOKncVY4Japg3G7oqOfIiLRKmaD3L+eP3j0nY7RhZf3bnb7fffdF1oIOzs7mylTpvDII49QUlJCWVkZV155JVOmTOGDDz7g9ddfJxAI4Pf7ue+++/B4PEd8frirr76aZ599lueff541a9aQmJhIWVkZkydP5pprrqGmpoaFCxfi8Xg4ePAgN998M8cdd9wR+/rBBx/w17/+ld69ezN9+nTOOeec0PEBli5disfj4eKLL+bSSy/ljDPOIDk5mYKCAvr27YvH42Hz5s3ce++9pKamts8PWEREOkV4Ne74KKnGhXMZw41TBmIMvBYMnCt3FmOt5dZp6QpzIiLNiNkgFylr167F7Xbzs5/9DICrrrqKyspKUlJSuPPOO/H7/Vx77bU8++yzDB48mJqaGgA+++wzNm3aRHV1daPnH67uOQAZGRnMnz8fgIsvvphrrrmG5557jilTpnDxxRdTXFzM/Pnzeeqpp47Y3+XLl3PJJZcwc+bMIx6/trY29OlsSUkJ3//+9/F4PNx8882cdtppzJo1ixdffJFly5bxzW9+sy0/OhER6USNqnEToqcaF85lDDdMHogB/hPs7zu7SrDkM09hTkSkSQpyrZSTk8Po0aNDj8ePH8+mTZvYuHEjDz/8MADx8fEA3HLLLfz0pz9lxIgRLF68mPLycnbv3t3o+c0ZPnx46OuEhAQANm3aRG1tLRs3bgQgJaXpT1jnzZvHn//8Z15//XVuuOEGsrKyGmyvu5wToHfv3ng8ntBrjRw5MvR1cXFxs/0UEZHo8uKGwgbVuK8Mjq5qXLhQmDPw6pdOmHt3VwnW5jPv9HQ8CnPSDVx66aWN2pYuXRqBnkisiNkgd7TLH5vi8Xjw+/3H/LojR47kpZdeCj3+6KOP+M53vkN8fDzXXXddg32NMYwYMYJAIMCqVas4//zzj/j8GTNmtKoPw4cPZ+jQoZx33nlH3Tc+Pp4bbriBPXv2sHDhQn7/+99TXl4e2r5u3TpOPfXUIz43Gj+5FRGRoyssr+G/Ww+FHkdrNS6cMYa5pwzEGMMrm4sAeC/bqczdpjAn3cDq1asj3QWJMTEb5CJl3LhxvPbaa9x+++0kJCSQnp5O7969WbVqFfPmzSM+Pp4pU6ZwySWXcPbZZ3P77bfj9/uZOHEixpgjPt/tduNyuXC5nElE4+LiAHC73bjd7tBr17Vfc8013H333SxbtgyXy8VXv/pVzjzzzCP295lnnmHDhg2UlJTwjW98A4Cvfe1r3HTTTfTv35+EhITQa9QdH2jQn8P7ISIi0e3FDYX4Y6QaF84Yw/UnD8AA/w6GuVXZTmVu/nSFORGRcMbaqJ3m1+bn5zdoqJsgpC3aWpGThtpjTOqkpaVRUFDQLseS9qExiV4am+gV6bEpLK9h7svbQ0Fu4axMJqX3iFh/joW1lj9+vI9/BcMcwGlDejD/9Azi3G0Pc5EeI2ladx6bjIyMRm15eXkR6EnTuvP4REp6ejrAEX/xqSLXRaxcubJRSX7gwIFce+21kemQiIhExNKwatwJabFTjQtnjOE7Jw/AGPjnJifMrckp5afv5vGj6e0T5kREYp2CXBcxY8aMVt9rJyIiXUtBeQ2vh90bNyeK1o1rLWMM/ztpAC5j+McXBwB4P1dhTrquF154IdJdkBgTU0Euii8D7bY0JiIi0ePFLlCNC2eM4dqv9AdoEOZ+8k4ed5yRTpzbFcnuibSradOmRboLEmNi6jegy+XS/W1RxO/3hyZEERGRyOpK1bhwdWHukhP7hto+zHPCXE1toJlnioh0bTFVkUtMTKSyspKqqqpj/s8pISGBqqqqdu5Z92OtxeVykZiYGOmuiIgIsHR9eDUuKearceGMMXzrpP4Y4MWNTmXuw7wyHlmZx50zMlSZE5FuKaaCnDGGpKSkNh1Ds+2IiEhXs7+shje2xda6ca1ljOGak/pjjGHphkIAPsov48fBMBevMCci3Yx+64mIiMS4hvfGJXHSoPZZFibaGGO4emIal43tF2r7OL+MH6/Io1qXWYpIN6MgJyIiEsO6QzUunDGGqyam4R1XH+bW7i7jYYU5EelmFORERERiWHepxoUzxnDlhDQuH18f5tbtLuOhFXlU+RXmRKR7iKl75ERERKTe4dW4K7t4NS6cE+b648Lwt8+de98/2V3GQytyWTAzkwSPPquW2PLYY481arvtttsi0BOJFQpyIiIiMSq8Gjc6LYmJ3aAad7g5E9LAwN8+c8Lcp3vKeXBFLvcozEmMefzxxxu1KchJc/QbTkREJAY51biDocdd/d645swZn8aVE9JCjz/bU86Db+fqMksR6dIU5ERERGKQU41zvu6u1bhwl49P46qJYWFubzkPvJ1LpcKciHRRCnIiIiIxRtW4I/OOS+Oaif1Djz9XmBORLkz3yImIiMSYpWHVuDH9VY0Ld+m4fmDgL5/sB2D93nIWv5XDvWcOISlOn19L9Jo3b16kuyAxRkFOREQkhuwvq2FZWDVuznhV4w536dh+uIBngmFuw74KFr+Vw32zFOYkemliE2kt/TYTERGJIarGtcwlY/tx7VfqL7PcuN8Jc+U1tRHslYhI+1GQExERiRGqxrXON0/sx/9OGhB67IS5XMqq/RHslYhI+2jRpZVer/fJ4L6pwJc+n2+R1+udDdwKlAG5Pp9vXnDfVrWLiIhIy7ywvr4ad6KqcS1y0Zi+APxp7T4AvthfwW3/2MDdZwwiOc4dya6JiLRJiypyPp/v+z6f7wafz3clMNzr9Z4A3AVc4vP5vEC51+s9x+v1mta0d8y3JCIi0vXsK63hze1h1TjNVNliF43py3Un11fmPt9dwqLlubrMUkRiWqsurfR6vb2ANKA3sNHn81UFN/0DmAUc38p2ERERaYHwe+NO7J/EhIGqxrXGhaMbhrnNBRUsWp5DWbXCnIjEphYFOa/XO8rr9f4V+Ah4AnADB8J2OQD0C/5pTbuIiIgchapx7ePC0X2Ze8rA0OPNBZUsXJ5DqcKciMSgFt0j5/P5tgJXeb1eD/A34NdA37Bd+gKFwT+taW/A6/XOBeYGX5O0tLQWfyMt5fF4OuS40nYam+ijMYleGpvo1RFj86dPt4aqcRPSe3LW2CwFuWP0P2lp9OzZk58t3wLAlsJKHly5m59/cxypCVqVKdK68++2FStWNGqbOXNmBHrStO48PtGoVb+xfD6f3+v1uoGdwDiv15sQvFzyYmAFsLWV7Ycf/yngqeBDW1BQcIzfVtPS0tLoiONK22lsoo/GJHppbKJXe4/NvtIaXtm4J/T4sjG9KCxs9FmotMI3xw+ktKSY3324F4Av9pZyk+8T7j9rCD0SNAFKJHXn323nnntuo7a8vLwI9KRp3Xl8IiU9Pb3JbUcNcl6vdxIwDygFUoAXfT7fLq/XuxhY4vV6y4DdwOs+n8+2pr2t35iIiEhXd/i9ceN1b1y7+OrxfQBCYW7rgUruW57DYoU5EYkRxlob6T40xebn57f7QfVJQvTS2EQfjUn00thEr/Ycm32lNdz4z23UBv+rfuDsIUwYlNIux+7Owsfov1sO8psP6iueI/smcP9ZWaQqzEVEd/7dlpGR0ahNFTkJVuSOeC29FgQXERGJUks3FIZCnKpxHeO843rz/VMHhR5vO1DFfW9mU1KlCVBEJLrprl4REZEotLe0mmXb6meqvEIzVXaYc0f1xgBPvr8HC2wvquLeN7NZfHYWPVWZk04yderUSHdBYoyCnIiISBQKr8aNHaBqXEc7Z1RvjIFfr3HC3I4ipzK3+Kwh9EzU2yXpeEuXLo10FyTG6NJKERGRKLO3tJo3tx0KPZ4zXtW4zjB7ZG9uPm1Q6GaUHUVV3PtmDsWV/oj2S0TkSBTkREREooyqcZFz9sje/GDq4FCY23mwinvezOGQwpyIRBkFORERkSiialzknTWiFz8MC3O7DlZx77IcDirMiUgUUZATERGJIi+sr6/GjRuQpOUGImTWiF7cMm0wrmCa23WoinuXZSvMiUjUUJATERGJEntLq1m+PawaNyEtgr2RM4c7lbm6MJd9qJp7lmVzsEJhTkQiT0FOREQkShxejRs/UNW4SDtzeC9unZYeCnM5h6pZsCybIoU5EYkwBTkREZEooGpc9JoxrGeDMJdb7FTmDijMiUgEaWEUERGRKNCgGjcwWdW4KDNjWE8M8PiqfAK2Psw9ODuLvkl6OyVtl5GR0agtLy8vAj2RWKGKnIiISIQ1qsaN7xfB3khTzhjWk/mn11fm8oqrWfBGNoXlNZHtmIh0SwpyIiIiEeZTNS5mnD60Jz+ano47GObyS5zKnMKciHQ2BTkREZEI2lNSzVuqxsWUaVk9+dH0jLAwV8OCZdkUKMyJSCdSkBMREYmgFzaoGheLpmal8qMz6sPc7pIaFryRzf4yhTkR6RwKciIiIhGyp6ThvXFXjNdMlbFk6pBU7jgjA0/w3dSe0hruWaYwJ8cmLy+v0R+R5ijIiYiIRMgLGwoJBKtx4wcmM25gcmQ7JK126pBUbleYE5EIUJATERGJgMOrcXNUjYtZp2Y2rswtWJbNvlKFORHpOApyIiIiEaBqXNcyJTOVO8/IxBNcm2BvMMztLa2OcM9EpKtSkBMREelkujeua5qc2YO7ZmSEwty+MucyS4U5EekICnIiIiKdzLe+vho3YWAyY1WN6zJOyejB3TMyiAuFOT8L3lCYE5H2pyAnIiLSiXaXVPPWDt0b15WdnNGDu2fWh7n95X7ufiObPSUKcyLSfhTkREREOtELqsZ1C5PSG4a5gnI/dy/LZrfCnIi0E0+kOyAiItJdNKrGTVA1riublN6DBWdm8vCKXKprLYXlfhYsy+ah2eWSf60AACAASURBVFkMTo2PdPckylx66aWN2pYuXRqBnkisUJATERHpJA3ujRuUzNgBqsZ1dV8ZnMKCmZk8FB7m3sjmwdlZpPdUmJN6q1evjnQXJMbo0koREZFOsLukmrd1b1y3dNLgFO45M5N4t3OZZWGFU5nLK9ZlliJy7BTkREREOoGqcd3bxEEp3BsW5g4Ew1xucVWEeyYisUpBTkREpIOpGicAEwalcN+sTBKCYa6ows89y3LIPaQwJyKtp3vkREREOpiqcVJn/MAU7ps1hMVv5VBVa4NhzrlnLrNXQqS7JxH0wgsvRLoLEmMU5ERERDrQ4dW4K1SN6/bGDUxm4awhLH47h0q/paiylgXBMDdEYa7bmjZtWqS7IDFGl1aKiIh0IN/6glA1buKgZE5UNU6AsQOTuW/WEBI9zmWWB4NhLluXWYpICynIiYiIdBCnGlcceqx74yTc2AFOZS7R47wdO1RZyz1vZJN9UGFORI5OQU5ERKSDqBonR3PigGQWzcqsD3NVtdyzLJtdCnMichQKciIiIh3g8Gqc7o2TpowZkMyiszJJOizM7SyqjHDPRCSaKciJiIh0gOc/r6/GnTQomTGqxkkzxvRP5v6zh5Ac57w1K66q5Z43cxTmRKRJCnIiIiLtLL+4mhU7dW+ctM4JaUksOqs+zJUEw9wOhTkROQItPyAiItLOwu+NUzVOWuOEtCTuP2sIi5bnUFYToKSqlnuXZbP47CxG9E2MdPekAz322GON2m677bYI9ERihYKciIhIO8opqmhYjZugapy0zvFpSdx/9hAWvhkMc9UB7n0zmwcU5rq0xx9/vFGbgpw0R5dWioiItKP/+yC7vho3OIUx/VWNk9Y7rp8T5lLinbdqpcEwt+2ALrMUEYeCnIiISDvJK67m9c37Q4/njO8Xwd5IrDuuXxKLz8qix2FhbmuhwpyIKMiJiIi0mwb3xqkaJ+1gVL9EFp9dH+bKqgPctzybLYUVEe6ZiERai+6R83q9TwMBoC/wss/ne9br9S4DtobtdqfP5zvo9XonAg8DpUA5MNfn89U01d6O34uIiEjE5BVXs7LBTJWqxkn7GNk3kQfOzuK+N7MpqQ5QVh1g4Zs5LDprCMenJUW6e9JO5s2bF+kuSIxpUZDz+XzXA3i9XhewEng22H7jEXZ/GLjG5/Md8Hq91wHXAk830y4iIhLzVI2TjjSir1OZC4W5mgALlzth7gSFuS5BE5tIa7X20sp4oDD4dYnX673P6/X+wev1fhvA6/UmAn6fz3cguM8/gFlNtbex7yIiIlEht7iqQTXuCq0bJx1gRN9EHpidRWqCG4DymgCLluewuUCXWYp0R61dfmAx8CiAz+f7JoDX6zXAk16vdwfwJXAwbP8DOJdj9m2ivQGv1zsXmBs8Pmlp7f8focfj6ZDjSttpbKKPxiR6aWyiy28+3hyqxp02rA/TxwyJbIekWbF8/qSlwa979+GHf/+cgxV+J8y9lcvPLx7LuME9I929NovlsekOND7RpcVBzuv13gqs8/l874W3+3w+6/V6XwEmAquBPmGb++KEtsIm2hvw+XxPAU8FH9qCgoKWdq/F0tLS6IjjSttpbKKPxiR6aWyiR25xFW+EzVT57SlDNDZRLtbPn94G7p+Vyb1v5lBcVUt5dS23vLSehWdlxvwlvbE+Nl2dxqfzpaenN7mtRZdWer3e7wLFPp/vb03sMgP4yOfzVQHxXq+3rtp2MbCiqfaWvLaIiEg0e+HzwlA17iuDU7pEVUSi37A+iTw4O4tewcssK/wBFi3P5Yt95RHumYh0lqNW5Lxe7zTgLuB1r9c7Ndh8N3AnkAIkAu+HVepuB/7o9XpLgCrgpqO0i4iIxKTc4ipW7gq7N26CLjmSzjO0dwIPzs7injezOVRZS6U/wKK3clg4awgnDojtypyIHJ2x1ka6D02x+fn57X5QlYSjl8Ym+mhMopfGJjo8/l4+K4KTnEwanMLCs4ZobGJAVxuj7ENV3Lssm4OVtQAkegz3zRrC2BgMc11tbLoajU/nC15aaY60rbWTnYiIiAiQe6iKd8KqcXNUjZMIyeoVrMwFw1yl37L4rRzuPXMI4wbGXpjrrlatWtWobdq0aRHoicQKBTkREZFj4Ftff2/cpMEpWstLImpIrwQeCoa5ovAwNyuT8QNTIt09aYHLLrusUVteXl4EeiKxorXryImIiHR7qsZJNMrslcCD52TRJ8n5nL6q1vLAW7l8tqcswj0TkY6gICciItJKz4dV405OVzVOokdmT6cy1zc8zL2tMCfSFSnIiYiItELuoSre2Vlfjbt8vKpxEl0yesbz0Ows+gXDXHUwzH2qMCfSpegeORERkVZ4fn0hdfM9qxon0Sq9ZzwPnZPFgjeyKazwU11refDtXBbMzOSkwbpnLhpNnTr16DuJhFGQExERaSFV4ySWDE4Nhrll2RSWO2HuoRW53D0zk68ozEWdpUuXRroLEmN0aaWIiEgLqRonsWZwqnOZZVpy/WWWD72dy9r80gj3TETaSkFORESkBXIOq8bNUTVOYkRdmOsfDHM1AcvDK/IU5kRinIKciIhIC/g+b1iNO17VOIkhg4KXWQ5IqQ9zD63I46M8hTmRWKUgJyIichQ5h68bp2qcxKCBPeJ5cHZ9mPMHLD9eqTAnEqsU5ERERI5C1TjpKgb2iOeh2UMZkBIH1Ie5D3MV5kRijYKciIhIM7IPq8ZdMUHVOIltA3rE8dDsLAb2qA9zj7yTywe5JRHumYi0hoKciIhIM3yfF4Sqcaekp3BcP1XjJPbVhblBoTAHP3knj/dzFOZEYoWCnIiISBOyD1Xx7q76N7ZzVI2TLqR/ShwPHiHMrVGYi4iMjIxGf0SaoyAnIiLSBFXjpKvrnxLHQ+fUh7laC4++k8dqhTmRqKcgJyIicgSqxkl3kZYcx8PnZJGeWh/mfvpOHquyi4/yTBGJJAU5ERGRI3g+rBo3OUPVOOna+iU7l1k2CHPv5vOewpxI1FKQExEROUz2wSreC6vGXa5146QbqA9z8QAELPzs3Xze26UwJxKNPJHugIiISLR5fr2qcdI99Ut27pm7Z1k2ecXVTph7L5+AhTOG9Yx097q0vLy8SHdBYowqciIiImFUjZPurm+Sh4dmZ5HZs74y9/iqfFbuVGVOJJooyImIiIRZonvjROiT5OHBw8Lcz1fls2LHoQj3TETqKMiJiIgE7TpYxarssJkqx/ePYG9EIqtPsDI3pFd9mPvF6t28rTAnEhUU5ERERIIazlTZg1H9EiPaH5FI6x2szGWFhblfKsyJRAUFOREREY5UjdO9cSIAvROdMDe0VwIQrMyt2s3y7QpzIpGkICciIoKqcSLN6ZXo4YHZQxja2wlzFvjVaoU5kUhSkBMRkW7v8GrcFRNUjRM5XK9EDw+ePYRhh4W5ZdsORrZjIt2U1pETEZFuL7waNyWzByP7qhonciQ9Ez08MDuL+97MZkdRFRb49Zo9AMwe2TuynYtxl156aaO2pUuXRqAnEisU5EREpFvbdbCK93RvnEiL9Uxws/jshmHuiTV7CFg4d5TC3LFavXp1pLsgMUaXVoqISLf2/OcFoa9VjRNpmZ4Jbh44O4sRfRJCbU++v4fXt+oyS5HOoiAnIiLdlqpxIscuNViZG9m3YZj77xaFOZHOoCAnIiLd1pKwatypqsaJtFpqgpvFZ2U1OHd+88Ee/vNlUQR7JdI96B45EWnE+v3Yfy3h4MEC7LTZmBPGRbpLIu1uZ1Gl1o0TaQc9EtwsPnsIi5bnsKWwEoDffbgXC1xwfJ/Idi6GvPDCC5HugsQYBTkRacQueQq74jWqAFYtx5w+G3PZtzEpqZHumki7eX59YejrUzN7MELVOJFj1iPezaKzGoa533+4F2vhaycozLXEtGnTIt0FiTG6tFJEGgi89Sp2xWsN2ux7ywjc+z0Ca97CWtvEM0Vih6pxIu2vR7yb+88awvH96j8Ueeqjvfx784EI9kqk61KQE5EQ+8Wn2CVPhR67+g+s31hyCPvHnxP4+X3YffkR6J1I+1nyuapxIh0hJViZOyGt/px6+qN9/GuTwpxIe1OQExEA7N58Ar/7CQQCTsPQUaQ9sQTX9++GPmHVii8+JbDwZgKv+LD+msh0VqQNdhZVsjpH1TiRjlIf5pJCbX/4eB//VJgTaVcKciKCLS8l8OsHoLzUaejdF9dNCzAJCZiTTsO1+NeY2d8AE/yV4a/B/uNZAg/cit26MXIdFzkGqsaJdLzkODeLzspkdFiY++PH+3j5C4U5kfaiICfSzdlALYGnfwZ78pyGuHhc31uA6d0vtI9JTMZ1+XW4FvwMskbWPzk/m8BP7iTwlyexZaWd3HOR1lM1TqTzJMe5WXhWJmP614e5P63dxz++KGzmWSLSUgpyIt2cXfp/sH5t6LG59geY4ccdcV8zdBSuu3+G8X4HEuqrGHblfwnc9z0CH76jyVAkqqkaJ9K5kuPc3DcrkxPDwtyf1+7npY0KcyJt1aLlB7xe79NAAOgLvOzz+Z71er2zgVuBMiDX5/PNC+7bqnYRiZzAu29g33g59Nhc4MU1ZUazzzFuN+aci7CTphH42+/h0w+cDcUHsU/9FLvqTVxX3ojpP6gjuy7Sajs6oRpn162h6MOVBIYdj5lxLiYxud1fozkH9vvZurmS1J5usobHk5Lq7tTXFzkSJ8wNYfFbOWzcXwHAM+v2g4VLxvY7yrO7j8cee6xR22233RaBnkisMK359Nzr9bqAlcAZwDLgAp/PV+X1eh8EVgTbWtzu8/neaOblbH5++8+Ml5aWRkFBQbsfV9pOY9O57JaNBB67B2r9TsNJp+H67p0YV32h/mhjYq2FdasJ/O0pOBh230N8PObCKzCzL8J4tFxlR9D50nqPrMxldY5zCfBpQ3pw14zMdj2+LS8j8KP/gepqpyE5BXPmBZizv47p2TnraL39n2JKigOhx/0GeMgaEc/gzDjcbtMpfYgFOn8io6ImwINv57B+X0Wo7ZqT+nNpWJjrzmOTkZHRqC0vLy8CPWladx6fSElPTwc44i/w1l5aGQ8UAscDG30+X1Ww/R/ArGNoF5EIsAV7Cfz2x/UhLnMYru/c2iDEtYQxBjNpGq77n8TM+hqY4O+Z6mrsi88QeGgedvvmdu69SOs51bj6+zgvH9cB1bhP1tSHOIDyMuyrLxC44zrnPtK9Hbtshw1YSksCDdoK9/lZt6acN/5ZzPq15RQfrO3QPog0JynOxb2zhjBuYH2l+i+f7OeF9QoGIseitR+VLwYeBfoB4dMOHQi2tba9Aa/XOxeYC+Dz+UhLa///aD0eT4ccV9pOY9M5AhXlFD30EwIlhwAwPXvT797HcA8Y3Gjflo9JGvxgATXnX0zxb3+Cf+dWpzl3J4FHbifp/EvocdUNuFJ6tON30r3pfGmdx9d8Efp6xsh+TDm+fatxAEWffkBdjDOJydjKcueBvwa78r/Yd14n4bSZpFx8NXHHn9jur19W6sfa4Hkd/Eyl7qKbmmrLji3V7NhSTdqABE4Y25Pho1KJi++et8rr/ImsX/5//fjRPzeyNtf59/rspwUkJ6fwP1OGaGwOE20/C41PdGlxkPN6vbcC63w+33ter/cEnPvl6vTFqdQVtrK9AZ/P9xRQtxqx7YjSrUrC0Utj0/FsIEDgt49AXdByezDfvZMiVxwc4Wff6jHpOxB7x6OYN/+J/edzTnXCWir+8yIVq9/CNWcuTJqKMbrEq610vrTcjqJKVmyr/y/nkhNS2/1nZ0uLCdTdLwqY+36Byd5O4LUXYeeW4E6WqtVvU7X6bThhPK7zLoFxk9rtfDhQ4A993bO3m8nTU8jZWU3O9mrKy+ordQX7qijYt5817+wnY0g8WSPi6d3P3a3OS50/kXfn6QN58O0aPtvrfODx1OpdlJaV8f1ZozU2YaLtZ6Fzp/MFL608opZOdvJdoNjn8/0t2LQVGOf1ehOCl0tejHMvXGvbRaQT2Zefg0/WhB6ba76HGdW+lQHj8WDOu8SZDOW539XPiHnwAIHfPQITp+C64gZMv/7t+roiTVnyef2bjqlDejC8T/vPVGnXrYFa57LFuOPHEug/CPoPwjVpKny5wQl06z+uf8Lmzwls/hwyhmLOuwQz+Yw2309aWV4f1hKTDUnJLo4/MZHjxiRQsM9P9rZq9uTVEAjuVuuH7B3VZO+oJrWXi6wRCWQOjSM+oXtW6aRzJXhc3HNmJg+tyOXTPU6Ye+6zApKSsvnGqM6dJChazJuneQCldY462YnX650GLAFeD2u+GxgL/ABnFsrdwO0+n896vd5ZrWlv5qU12Uk3o7HpWIH3V2D/UD8jljnnIlze7zT7nLaOibUW+9G72CVPQ/HB+g0JiZiLrsKc9XWMW7PqHQudLy2z/UAlt/5nZ+jxLy4Y1iFBrvbxe+GLTwHo8b8/pGLq2Y32sbk7sP/9B/bDlaHQF9I3DXPORZjp52ISkxo9tyW2bapk46eVAAwbFc/4kxu/Ga6qCpC3s5pd26spLQ402u5ywaDMOIaOiKffAE+XrdLp/IkeVf4AD6/I5ZNgmAOYM74fV0zQh33RSOdO52tuspNWzVrZyRTkuhmNTcexO74k8Ohd4K9xGsZNwnXzvRhX8yGqvcbElpdiX/x/2JWvNdyQNRLXt76PGTqqza/R3eh8aZkfr8xlTXCSk6lDenBnO89UCWCLiwjM/zbYABhD2tP/oMg2HYBs4X7sspex77wOVZUNNyb3wMy6wPmQo2fvVvVj/dpydmxx7tIbMyGRUWOaDqzWWooKa8neXk1+dnWjXAmQnOIia0Q8Q4bHk5jUtap0On+iS5U/wMMr8/hkd1mo7fLx/bhifFqX/TAhVunc6XzNBTn3okWLOrUzrbCopKTk6Hu1UnJyMuXl5UffUTqdxqZj2KJCAo/dCxXB/yAHZeL64SJMfMJRn9teY2Li4jETJ2NOPAm740sITrTCoSLsO29AWQmMGoPxxLX5tboLnS9Ht/1AJX/8eF/o8Y+mp9M7qf2Xw7Cr3oLPP3QeHHciqRdd0ezYmOQUzLhJmJnnQ2Iy5O2C6uCkzjXVsGUD9q1XoKgABmVgUlJb1I/wKlvW8Hh69m76gxpjnEsvB2XEMey4BJJTXFRVWior6j/cramxFOzzs+PLKg4e8OP2GJJ7uLrEG2udP9HF4zKcnpXKtsJKdpc6Hzhu2FdBwML4gcld4t9cV6Fzp/OlpqYC3H+kbQpyEjU0Nu3PVlcR+MVC2Btchya5B675D2J6t2wB1vYeE9O3P+aMcyAuHrZtgkAtYGHHl9g1b2P6D8QMav+KSVek8+XofvvBHvKKnQrV1CGpfH1036M849gElv4JCvcDYM6/hJSxJ7VobEx8Aub4sc7SHX3SYE8ulAeXSAjUwq6t2LdedYJe2sCjnrfbNlVRVekEseHHO+GsJdxuQ+++HoaOTGBQRhwuF5SVBEL30oHzOD+7huzt1VRXW5JTXMTH8IyXOn+ij9tlmJaVSk5JgNxDTqV6w74KagMKc9FE507nU5ALo3+A0Utj076stdg//Rw2fuI0uFy4broHM+y4Fh+jI8bEuNzOm9fJ07G7c6Bgr7Ohshz74TvY3B2YUSdikrrnze4tpfOledsPVPLHtZ1QjTtYiH3+D84D48J17c2k9OnXqrExbg9m2HHOJZUZQ7EFe+FQ3Yo9FnbnYN95Hfvlemdh8f6DjvimdvP6ytDSkMePTSQuvvVvfBOTXAwYHMfw4xLo0dNNTY2lImzGy1o/HCioZceWag4U+HG5IKWHC5crtt5k6/yJTm6X4YIJWazPO8DuEqcyt3F/Bf4ATFCYiwo6dzpfc0Gu/f9XE5GoYF99AfvhO6HHZs5czJiJEexRQ2ZAOq5bF2PfX4H1/bH+cst1awh88Snm4msws7561Pv4RI6k4UyVqQzrgAlOAOxH79Uv1jZ6vBO0jpFxueGU6bhOPt2Z1fK1F2HDuvod6ma6zBzmzHR5yvTQTJe1tTZUjcNAYlLb3vC6PYbMYfFkDountKSWnB3V5Oyorn8NoGCvn4K9fuLinX2HjogntZfOV2mbBI+Lu2Zk8MjKPD7Kd24JWLqhkIC1fOuk/gpzImFUkZOoobFpP3btauyzvwk9Nmd+Fdc3rmz1cTp6TIwxmMxhmOnnQGkJZG93Nvj9sP5j7IZ1mKGjML2O/c1xV6XzpWmdVY0DCDz/Byhy1qgzX70UM3RUm8fGGINJG4jrtFmYr5wGlRWwO6c+MBYfhHWrsauXOyt/ZwylosoVmugkMckwanT7Bdf4BBf9B8Yx/PgEevVx4/dbysKqdIFaOFhYy86t1ezf41RRUnq4cbmj9w23zp/olZycTFVlBdOyUtlRVEV+ifPv+ov9FVTXWiYOUmUuknTudD5dWhlG/wCjl8amfdicHQSeWFw/vfkJ43FddxvG1fr7WTprTEx8AuakUzGjx2O3b4bSYmfDwULsu687b2RHjWnzOltdic6XpnXWvXG2cB/2hT87D9xuXP9zMyY+oV3HxvTqg5k0DTPtLCfI5e2qP7crymHDWuyK1yiuTSG3dggAqb3cZI04+mRGre6LMaT2dJM5NJ6s4fHExRvKS2tDk+ECVFZY9ub72bGlivKyAAmJhsQkE3VvvHX+RK+6sXG7DFOH9GDnwarQ+fzF/goq/ZaTumiYW7VqFTk5OQ3+DBkyJNLdakDnTufTpZUi3YQtPkjg1w/Wz4DXfxCuG++ImQBkjh+H675fYl97Efuqz6nMBQLY1/+O/fg9XFfdiBl/SqS7KVFs+4FK3s8tDT2eM75lE/scC/vRu/UPxkzE9OjZYa9l+g3AzLke+/XLsW+/in3z3/UfeJSVULH2Uxg3FYBEVxXQspkuj1VSsovjxyZy3IkJFOz1k729mt15Ndjwxca3V5O93VlsfOiIBDK02Li0Upzbxe3TM/jpu3mh8/ofXxzAWsu3Jw3ocmHusssua9SWl5cXgZ5IrNBvVJEuwtbUEPjtj+GAM3seiUnO5CYd+OayI5i4OFwXzsG18Fdwwvj6DYX7CPxqMYHf/QR78EDTB5BuLfzeuGlZHXdvHID9sD7ImVPO6LDXCWd69MT19Tm4Hvkj5qobof8gACoS66uOiR8tI/D7R7G7tnZ8f4yh/6A4Tp6WwjkX9uTEkxLpkdrwrUXJoQDr11Xwxj+LWbu6jIJ9NUTxGrYSZeLchh9Nz+C0IT1CbS9vKuJPa/fp35F0e7HxMb2INMtai/3rb2DrF06DMbjm/giTnhXZjrWBGZSJ67YHsauXY1/4k3MPHWA/fg+78RPMJd/CzDjvmC4Zla7p8Grc5eM6sBq3Lx/qgpLHg/nKqR32WkdiEhIwZ16APeM87NrVVK6tCG1LrCzEbnnXqRiOmYjrvEvgxJM6vHqRkOhi5AmJjDg+gaICZ7HxvJxqZ5URIBCAvOwa8rJrSOnhYsiIeIYM63qLjUv7qwtzP3s3j9U5zjn+z01FWAvfObnrVeZEWkpBTqQLsG+8jH3vzdBjc+m1XeISRGMMZtrZ2PGTsS/8yZncAaCiDPvX32LXvIXrmu9jMoZGtqMSFSJVjWPsJExyj6Z37kDG7cZMnk5lRSnsdtYeSKwqrN/hi08JfPEpDBleP9Olu2NnljTG0Le/h779PYz9ShJ52c4lloeKakP7lJUG2PRZJZs/r2RAuoehIxLoP8gTc8sYSOfxuAzzp2fws3fzWZ3jfLD3r81FWOA6hTnpphTkRGKc/fxj7NL/Cz02087GnHNx5DrUAUxqT8z/3oKdOovAs7+FffnOhm2bCDxwC+bcb2K+fjkmvv0neJDYsK0Tq3FAw6U9JnfOZZXNqayon0Uy+cr/waxKdipydat65+zA/uEx7N//gjnnYsz02ZiEjgu6deLiDcNGJTBsVAKHipx76XJ3VYcmSLEW9ub52ZvnJzHJMGR4PFkj4klO0TIG0pgT5tJ5/L183st2wty/NxdhreX6UwbGfJibOnVqpLsgMUZBTiSG2d05BJ7+KaEZBkaOxlz9vZj/z6wpZsxEXIt+5ayR958XnRkVamux/1mK/ehdXFd9FzP2K5HupkRAeDXu9I6uxu3OcWaPBIiLx0yc3GGv1VIV5fX3CiWPGIJr7HzsxVdjl/3Tmfm12pn1j8J92CVPYf/9N8ysr2FmfR2T2jn30fbq42H8yR7GTExid24N2durOLC/vkpXWWHZsrGKLRurSBvoYeiIeAZmxOGO4mUMpPN5XIZ5p6cD9WHulS8PErBww+TYDnNLly6NdBckxijIicQoW1pM4IkHnCnIAfqm4freXZi4uMh2rIOZuHjMRVdhJ59B4C+/ga0bnQ379xD4xULMlJmYy/+3TQszS2zZdqCSD8KrcePTOvT1wqtxjD8Fk5jcoa93NH6/pabaCXLGBQmJzhtZ038Q5oq52K/Pwb71Cvatf4fuNaW0BPuvJdj/voQ5fbZTpQtOnNLRPB7DkGHO/XGlxbVkBxcbr65qvNh4fIJxljvQYuMSxuMy3HZ6Oi6Tzzu7nH/T/9lyEIC5kwfiiuEwJ9IaWkdOoobGpuWs30/gyYcge5vTEJ+A69YHMAPS2/V1onlMTGovZ22tvmmwZQPUBK/VytuFfXcZ9Eh17gvqov+hh4+N3x98E9/C77Wm1uLuQvciha8bd3pWKl87oeNCvDOx0O9CU/+7Lrqy0aRCnX3elJcG2LnV+f6Tkp0JR8KZhATMCeMxs74OvftCfg5UlDkba2th5xbs8lecRcf7D8L06ph1944kPsFF/0FxjDgubLHx0vrLRGtroejwxcZT3W2+ly6af7d1dy0dG5cxnJqZyu6SGnYdcpbc2XqgkoMVtZyckdJlf/dHms6dzqd15ES6GPv8H2DTZ6HHru/ciskaEcEeRYZxuTBnnIudOBn7/J+wH6xwNpSXUNLfzAAAIABJREFUYv/fr7Fv/gsz86uY087EJEW2atIRrLV8uaGKLzdW4jLQo6ebnr1d9OztDv1JCFu3y1rL0x/v45XNRZw5vCe3TB0c8292wqtxho6vxpG7E/bkOl8nJMK4yE8qVFFeH3ySkpseT5OQgJn1NeyM853ZX//7EmRvdzbaAPbDd5xq44knOTNdjpnYaf8+XG7D4Mx4BmfGU14WIGdHNTk7qhpcMlpUWEtRYQUb1lWQEazS9erjjvl/w3Ls3C7DLdMGg4GVO50PV/671anM3ThFlTnp+hTkRGJM4O1XsW+/GnpsLroKM2laBHsUeaZnH8z1t2GnnUXgr7+F/XucDXm7sM/9DvviM5jTZmLO/Comc3hkO9tOrLV88Wkl2zY7n0QHLBQfrKX4YC1QE9ovIdGEQt2WsgpW7SjBAG/vKObsEb2YMCglMt9AO/nbZw1nqhzau2MnvGkwycnEKZiEyE+w0yDItWAqf+N2Y6bMwE4+A774hMBrL8EXn9bvsPETAhs/gawRzkyXJ5/e4TNdhktOcXHCuESOPzGB/cHFxvfk1VC3ZJjfD7u2VbNrWzU9e7vIqltsPF7LGHRHbpfhlqmDcQFvh4W5gLV879RBCnPSpenSSokaGpujs198iv3DY9S9ozGTz8DMub7DPpGOtTExAwZjpp/r/HxytjuToYDz966t2BWvYTeuA48HBmZ06pvT9mSt5fO15Xy5seyo+9b6nUvvigpqMcWGE13JTDQpDHMlsntPDT2Nm1q/xeMxeDyx9YZna2Elf163D3CqcfOnZ9A7seM+n7TWYp/9DZQ7FUDXxVdhBmU22q+zz5u9+X4K9zn/1gcOjqP/oJbdJ2uMwfQfjGvqWZiJU5z7bXfnAsHEdKgI1q7Crnkb3G5IH4rxdN7nv8YYUlLdpGfFM3RkPAmJhvLyQOh+QICqSsu+3X52bKmitLiW+HgXScnmqL8TY+13W3dyLGPjMoYpmT3YW1bDzoPOh1vbi6ooLPczOaOHqrbtSOdO52vu0koFOYkaGpvm2X35BH6+EKqd/6QYOgrX9xZ06BurWBwT4/FgxkzEnHmBc/9c4X4oOVS/Q1EBrFuDXfGac59T/0GYlMisAXYsrLV8/nEFWzfVj8ugjDimndWDAelx/P/snXd4HNd1t98729B7x6Kzd4pdEos6JVuSVQz32HESWf5cE8eWexwXWVZkJ45rnDiOuw1bvVCFolgkNolN7A1E7wBRFtt37vfHXewCBAiiLIAFOO/z6BF3dnYwwN2Zueeec36/lFQTMbEaQgOvV4ayGP3RhCBOmIgLmGht8lNX7eP8aQ/V5z20Nvvp7grgcasPWqxXnhRPFT/d30RDz+T0xgFqMeDFJ9S/Y+MQH/zEkIsBk33d1FWHPdryCq2kpo/+niBS0lTmbe0m0APQUK0a1ACcvXD0AHLnS+DzQn7hpFt9mM2CtAwzxbOsZGRbkFLi6NFD328poadLp7bKS321j0BAEp+oXXZxYjre264Wxjo2mhCszk+g1enjwsVwMNfW62eV3QjmIoVx7Uw+Ro+cgcE0Rzp70X/8nVAmgOQ0tE98JSrKuqIVERev+oE23QFnjyO3b0Ee3BPO0jm6kS89gXz5SVi4HG3j7bBkJUKL3iyd1CVH3nJRe8Eb2pZXYGH52jg0TWCL0cjIMg/Yv7Pbz3/tbMHbK0nHTLpmIZ6hf0ePW9La5Ke1yR/aJjRITAr23SX3672LmdoytnPtbt6sn8TeOAaagItla6NGIXZgj9z4xkVk5iDe/yDyzvchtz2PfO156O1TuuxGPvMH5IuPI66/BXHL3YiM7HH9vFGfnxCkZ5pJzzSzaLlOfbWP6kpvsKRY0evQOfm2m1NH3WTnWSgss5KVbUbMIIEfg6ExaYJPrskFBNsq1QLeq5VdSOCTa3JmlMiTgQEYgZyBQdQj9QD6fz+mFOUAzBa0T3wZkTqxhsczBSEEzFmEmLMI2X0R+fpWlY3raFU7SAnHDqIfOwhpmYgNt6lJanJ02RfouuTwfif11eH+t/wiC8tWx11ewU/A7061sSdY3WAS8O0bCrFIwWNbG0gTFtKFmWtSE3A5dPTA4ENIHbo7dbo7dYbsvesX3CUkamiT5PnV3zduUnrjdB35VnSZgPcxUrGT0SASkxF3vx+5+V51zbzyFLSrMla8HuS255DbX0CsvF710U2B2JLFqlE820bxbBudHaqXrr5moNl4U72PpnofMbGCwlIrBSU24uKNXrqZjEkTfGptDpqAredVMLetsgspJZ9amxvVwVx+fv6gbfX19VNwJpGnpztAfbWXni6dwlIr2XnRsRA23TECOQODKEc+/ms4diD0Wnzk04iSOVN4RtMXkZSKuOPdyM33wtGD6Du2qL9tX31WRyvyqd8hn/0jYvk6VZ45Z+GUl+TouuTQXicNteFAava8ROYu1obNMrx4tjM0kQH4uxXZLMhS6p25uVYONfaChLhUjY/fmk1vrx4STOn7r79qYH8um71LHKiaORHZu7PtrgHZuPdOQjaOytPQEQwe4xNh/tKJ/5kjQEo5IJCLGWdG7lKELQZx0zuRm25HvvW6Ki2tu6De1HXk/p3I/TthwXK0zffCvCVTcr2kpJlJSTOzYFksjbVBs/G2gWbjZ457OHPcQ2aOmUVLY4hLlJO28GAwuWhC8Ik1yhex7x742oVuJPDpKA/mZhIet059jY+6qnD5N0BLo4+b70ya8sqOmYARyBkYRDH6G68iX34q9Frc8W60NRun8IxmBkIzwdJVmJauQrY2IXe+hHz9lZA3GIGAmrS+9TrkFigLg3U3IOImX+FRD0gO7HHSVB8O4orKrFx3Yxbt7e2X/dypVhf/c6A59PqGkiTumJMSen3vgjQVyKFKj967JIO0RDMJiSbyCsLH8Xl1ursGBng9XYFQ+1R/pI7at0uH6oHZu8R+mbukZBOJSWPP3v25XzbuuqJECic4Gweo70IQcc26SRX9GA6/T4aqhTUTWK0TM0EVJhNizUbk6g1K1fKlS5UuD6GfOARFs1SG7pp1UyImZDYLCkqsFJRY6ekOUFPppa5qoNl4a5Of15qasNqCxuSlVhKTorek2mBs9AVzAnglGMxtv9CNlPCZdUYwN1H4/ZKmOh911V7amv1D9mnrOlRXepmzIGbwmwajIjqeRAYGBoOQ504gf/eT8IZlaxB3f2DqTmiGIjJzEPd9GHnX+5EHdyO3b4FzJ8I7NNYi//QL5BO/RqzZqIK6orJJObdAQPLWG720NIazXiWzrSxcHjts1uOiy88ju+rxBxM1ZWk2Pr46Z8BnFmfHMTs9hrPtbvy65NlTHXx4edagY1msGumZGumZA3vvBmXvunRcvfqgz4PK3nncftqa+2XvBCQkXZK9S1ZCLcOhsnEqABXAexZNQm+cHkC+9UbotVh5/YT/zJHSP2MaG6dNeDZMCAELl2NauBwZFH+RB3arKB6UIMwvHkVm5iBueRfiupsmXRilj8QkEwuXxTJ/cQxNDT5qKr0DMshej+T8aQ/nT3tIyzBRWGojt8Ay7dRbDS6PJkTIgqDPX25HlcrMfdYI5iKGrkvamv3UVSurkIB/8D6aBkkpJjo71Cpg9TkPs+bZLt8aYDAijEDOwCAKke0t6D/9rjJMAsgvUqbfmlGGMFEIiwWxZiOs2Yisq0Lu2ILcsx08LrWD14Pc9TJy18tQMkd50q28fsImqX6/CuL6TzzL5tqYvzRm2Mm6LyB5dFc9F13qc4k2E19cb8dmHvjdEUJw38J0Htmp+i9ePNvJ/QvTibdeOTMhNEFCommI7J2ku2tg5q67KzDkQ71PZbCnSx/Q92e1iQGBXVKKRkKSCVMwezcV2TjOnoCuDvXvxGSYu3jif+YIiaTQyWgRRbMQH/sCsqUR+crTyDe2KlVLgNYm5eH47B8RN74DsekORELSpJ5fH5pJkFdgJa/AirM3QO0FL3XVAZyO8Bezoy1AR5uTY4cgv1CZjaekGVOkmYAmBA+uzkYIdZ8DZR4upeQz6/KwGOW1Y0JKSdfFAHXVPhpqvCGl40tJyzRhL7KSW2DBZBK8+lw3HrfE7VKZu7xC6ySf+czCuEsZGEQZ0u1SCpV9kvkJSWif/CoiJm5qT+wqQtiLER/4OPK+DyP37kDu2AJ1VeEdLpxBXjiDrPhflXHYuBmRlRexn+/3S/bv6g15gwHMXmBj7qLhgziAXx1s5kSrCj41Af98XR5ZCUM3la+xJ5CfZKW+24vTp7MlGMyNFYs1rCjYh5QSp0OnKxjYdXUG6OnUcV4me+f1qJXdobJ3IhY8DZAvrFyUfsoXTY7gzwAT8Ek2x74SozUDnwhEVi7iAw8i73wv8rXnkdueDyvs9nQhn/4DcsvjiPW3KqXL9MGZ38kiLt7E3EWxrNuQzsljTdRUemlu6Gc27gubjSenmigstZJfaMUyQSWrBpODJgQfW5WNALYEg7ld1T10uWt5aEM+CSNYwJoMpoOwSU+3jzMn3NRXeXH0DH0fT0jSsBdZyS+yDhIXKiqzcuZ40B7irMcI5MaJkEMVr0YHsqGhIeIHzcjIoK2t7co7Gkw6xtgoZTz954/Aob1qg8mM9k/fQsxZOCXnY4yJQkoJ508qC4MDb4Qzpf1ZsExZGCxdPa6Jvt8n2bfTMUCoYe6iGOYsHNhLMNTYbKvs4od7GkOvP7wsk3uvEJhtPd/Jj/Y2AZASY+IXd5cNyt5NBD6vVBm7zkA4i3eZ7N3lCGXvQv13A7N3kUAGAuif/0hoYUX7/MOIOYuG/cxkXjenjro4e0JNimYvsDFvceyk/NzhkG4X8vVXkK88HVaH7UPTEKvWIzbfi7CXTM0JMnCM3C6duiovNZVeeh2DJ6aaSdl8FJbaSMswTbn40UxnIq8fKSW/eKuZF850hrYVJFv52iY72QlGQHE5vB6dhlof9dXeAc+m/thiBPmFVuzFFpJSLn+duF06W5/tDi2erL8lwch+X4G8vDxQ3QSDMP5yBgZRhHz2j+EgDhAfeHDKgjiDMEIImLUAMWsB8j1/j3wjaGHQFhYT4cRh9BOHISVdWRisvwWRMrqMkc+rs29nLxfbww/K+UtimDX/yg3h5zvc/Gx/U+j1dYWJ3LMg7Yqf21iczB+OtNHu8tPpDrCtsovb50y89YLFKkjLNJM2RPYuXJ6p+vBGnb0bUjlzjMbmp98OZ8dT0mDWgtEfYwKZytLKyyFiYhE334XcdIcSDXrpiXBGW9eR+3Yg9+2ARdeg3XYvzF08pcFRTKzGrPkxlM2z0d4aoKbSQ2OdL2THoQegrspHXZWP+ESNolIr9mKrobg3DRFC8MDKbFJjzPz+bRUs1nZ5+cJL1Xx1k53Z6VO/EBItBAKSlkb1vW9u9IXaYPtjMkOu3YK9yEpG1si8GmNiNfIKLNTXqJL6qrNelq0xwpGxYvrGN74x1edwOb7RE/Q+iiSGI330crWPjf7mLuQffxF6LW6+C+2Od0/hGRljMhTCFoOYtQBx4zsRpXORbhe0NALB5UW3C04fRb76LLKuCpGQCBnZV5yoej06e3f0hhrBARYsi2HWvKGDuP5j0+328/VXa+j2qCdtQbKVr24qwGK68kSzr9m/T8GyrtvLHXNS0aZgYi2EwGrTSEwykZFlIb/QSukcG6VzbbzQ1MGpXhe96CTYNGKENuTEAlSA19Ol09bsp77aR+VpD1XnvLQ2+unuDOBxqw9arOKKjfbyhb9CTaU6v+tuRlu88oq/x2ReNxfOekMiM8WzbMQnRkeJGIDQNFWmvHGzulY6O8JedAAtjcg925BHDyDiEyAnHyEmJzgaaoyEEMTFa+TarRTPshIbp+F26nj6KV76vJLWZj+VZz10dwYwW9RnjCxd5Jjo60cIwcLsOPISLbxZ34suwe2XbL/QTWGyDXvy1IjzRANSSjpaA5w94ebIfhd1VT5VPtmveE8IyMo1M3dRDEtXxZFfaCU+YXSZ6phYjZoLqp/W0R2gqMxqiAwNQ2JiIsC/DvWeEQIbGEQBsuos8lc/DG9YuBxx/99O3QkZXBGhabB4BabFK5DtLcrCYNfL4eyNrsPB3egHd6sJ6sbNiHU3qQnrJXjcOnt3OIKm24rF18RSPPvKE4qALnnsjQZaelVWKs6i8aUNdmItI58Q3zorhYpjbTi8Ok0OH7trelhfPDXCFENR1e1mZ6ta2BPSxXtvLKEgyYpzgHLmCLJ3LX7aWobO3iUOUM5U2Tvp9yEP7g7vH0Um4H24ozAjdylCCFi0AtOiFcgLZ9FfehwO7gn7N1adRf/59yAzB3HrPYhrb5wypcs+rFaNktk2imdZ6bqobAzqq72hqmqpQ2Odj8Y6H7FxgoISG4Wl1qgdA4PBbCxJJiPewnd31NHj1fEGJI/srOdvr8nirnmpV1Vw3tMVoK5afccv5x2akmYiv8jK4mXZ9Do7h9xnpKSkm0hJUwqWug41lV5mG1YEY8LIyBlEDVfr2MjOdvTvfzUsDpCTj/bZbyBsU39Tu1rHZLSIuHjE/KWIm+6E/CLo6R6YeXD0wPFDyG3PQmsTpKSFyi7dLp292x30dIUn5EtWxlI8a/iJbN/Y/PZwK9svdIe2P7Q+n3mZoysPspgEnoDO8RYlktLo8HLbrJSomcj8ZF8TjT2qDGdDURK3z1GTLKt1cPaubK6N7DwLKWkmYuM0NA18Pjny7N0ZDxfOemlp8tNd2YS7+SIIgSUpHtP9Hx7R32SyrhspJSffdofioXlLYiPaHzgRiNR0tJXXK4XYgB/qq9WiB6h74NG31IKI3w/5hRMW0I10jIQQxMRqZOdZKJltIz5Bw+tRint9+H3Q3uqn8oyHzg4/JhPEJxhZurEymc+drHgLawoSOdDgwOFV38NDjb10ewIsz42fksqEycLt0qmu9HD0gIvTx9x0tAXw+wbuExevUTzbytKVccxeEENqupnk5IRxj48QAs0kQv6ojp4AJbNtxjVzGYbLyBliJwZRw9U4NtLrQf+3L0PVWbUhLgHty48hsiOngDgersYxiRSyvkZZGOx9DVxDPPSKZuFZfxd7HcvpdQTvwwKWrYqjoOTKTfcZGRk8faCSR18P3yffuzid9y3JHNP5drv9/N1T5/EG1Ln8yw12rskbnD2cbM60ufj8S9WA6vT+0TtLKBhl6ZOUsl/2LtyD5xxC2OJyCHTik8yXWCOEs3f9mazrxuPWeflpFcSbLXD7vSlX+ET0IbsvIl99Hrn9eXD2DnzTFqOULm++G5E+tu/15RjvGPV0qSxdbZUXn3fwPMpqU8bkhaVWEqKo3HU6MBXPnS63n4d31HOqzRXatjIvnn++Pn9U1Q3Rjt8naewz627xDyiZ7MNiFeQVWLAXW0lNH1wyGanxCQQkW5/txhssXV5xbRx5BYbgzFAMJ3ZiZOQMooarbWyklKqc8vhBtUHT0D7xFUTJ7Kk9sX5cbWMSSURSMmLxSsQN74CMbLjYBt3hchSXW7A39nacMl7tj+SatfHYi0f2IGtySb764rmQ6feq/PhBpt+jwWbW6HQHONvuBqDd5eem0uQxHSuSXJqN2zwGIZaB2TvzkNm7uHgNoYHfK0MJokuOMjB7VzMwe9d10Y/bpSMlJCXH4Xa7hjpIRHF0q2ACVAZoJKW40YawxSLmL0HccAckJUNjbXjhI+CHytPI156H5kbIykUkRSZYHe+9zRajkZVroWSOjaRkEz6fHFDWGwjAxbYAVWe9tLX4EEKQkKAZ5scjYCqeOzFmjY0lSTR0e6ntUtdUQ4+Pgw0OVuUnEGeZvsG4rktaGv2cOebm8JtOGut8g0rQNQ1y7BbmL4llyYpYcuyqTHio50mkxkfTBAG/pL1V9YV7XDqFpdPvHjYZGBm5fhgZhujlahsb/YW/IJ/8bei1eN8DaDe+cwrPaDBX25hMJFJKNSndsYXeo6fYt/SfccWqLIPQ/Sw79lNy07xom26HpWsQ5su3MDu8Ab74Si21nSroyk208Njm4nF7IbU4fHzsmfPowcfCo7cVMTdj6lTcIpGNGy1SSly9Ol2HTtK1ax89CQV0p5TitGWM+BhCQHyi1s8W4fLZu/HQVO/jzddVFiszx8zajVOfQR0v0u9HvrlLKV3WVw/eYfFKpXQ5Z+G4/pYTcW9zOgLUXPBSe8E7oPSyD7MF7EUqS5ecakgUXI6pfO7oUvL7I2389Xh7aFt6nJmvb7JTnDrx7Q7333//oG1//etfR30cKSWdHQHqq73U1/hCWa9LSc8yYy+ykGsfuVdiJMfnUiuCDbcmGNfGEBj2AwYGUYY8vHdgELdhs8rcGMxYhBBQNg9n1mz2pvfgVjEYmu5j+ds/JrvtELSAfuptSE5TJWXrb0WkDQwgdCn5j90NoSAuxiz40gZ7RAxtsxIsbChOCvXcPX68nS9vtI/7uGPlT0fDk4X1RUkTHsRBULkwwUTMyS1kXVBG4OKOd6O/84PK9y5kjaD+fWlPCSgND0e3jqNbeS/1YbEKkpIHWiMkJpkwjVGtLRqtB8aLMJsR625Art0Exw6gv/gEnDkW3uHoW+hH34KSOSqgW74GoUVHtiQuwcS8xbHMWRhDa5Of6koPLQ3+AWbjVee8VJ3rZzZeZMViMbJ00YImBB9alkl2goWf7W9Cl9Du9PPFl2v4wvq8CS8337Nnz7g+3+sIUF/to65qaE9EgMTksFn3VN83LrUiuHDWy7LVRmgyGoy/loHBJCPrLqD/zw/CG+YuRrzvAaPJ9yqgpzvAntccePqCOA1WFrWT0WGDdo2QIkdXB/K5PyFfqIAlq9E23Q7zlyI0jYqj7bxZH+4l+tTaXIpSIhfg3LsgPRTI7atzUNvlmZQA6lJOt7k40KB+TwG8Z/HoPPnGg/S4kUf2h16LVesxWwSpGWZSMwb63rmcekgxs++/y02gfF5VRtRXSqQODgkJg33vRpK9m4mBXB9CCFi8EtPilcjK0+gvPQmH+ildXjiD/vNHICsPcdu7EOtuRFiio79G0wTZeRay8yy4XTq1QbPx/j2ZXRcDHD3g4sRhF3kFKkuXapiNRw23zkohK97C93bV4/TpuPw639pex8dWZbN59sT7bI4Gj0ensUb1vfX3IO1PTKwgv8iKvchKUkp0LHz0UTzbFgrk6qu9zF8ag802s+5nE4kRyBkYTCKyuxP9x98hNJPPzEF78KFhy+gMZgbdnQH2bHeESlw0E6xeH09m9jxY+2VkRyty18tKsa/rovqQrsPhveiH90JWLm+tvZ8/dheEjnnP/DSuL4qsTUBRio1V+Qm8Wa9UVJ880cGn1+VG9GeMhD/3z8YVJ02qt5N8+y3wetSL3AKlRDoEynfMRFy8iZx8S2h7cnIaVZUtAzJ33Z1DZ++Q4OjRcfRcPnuXmGwiOcVEQrJpgNfSQOuBmRsAiNK5mD7+RWRTPfKVp5C7txH6Y7Y0IH/7U+TTf1DejpvuGNLiY6qIidWYPT+GWfNstLf4qan0KrPx4NAFAlBbpURTEpI0CkvVZNswG596luXG88itRXzztVranH50CT/b30yzw8eHlmVOqaJlwC9pblDBW0tjOOvbH7MZcgus2IsspGeOzKx7KkhNN5GcaqLrYtCK4LxhRTAajNmjgcEkIf0+9J89Epalj4lF+8RXEQnR49dlMDF0XfSzZ3tvSN3OZIbV6xPIyArfgkVaJuLuDyDf8R44sg99+xY49Xbo/YYeH//RkRG6a1+TbuaDS0fetzUa7luYFgrkdlR18b4lGWTGW67wqcgxKBu3aPKycQDyrV2hf4uV1486S2KxaKSmm0lNvzR7JwcFd709o8vexffL3nVdDL8XM8MyckMhcvIRH/oE8q73I199Frl9C7iC2enuTuRTv0NueRyx4VbEzXch0iKrdDkehBBkZFvIyLbg9ejUVfuoqfQMsB1xdOucOOzm5NtucvMtFJZaycg2G1m6KaQoxca/bS7m29trOd+hFneeONFBk8PHZ9flYjNP3nUndUl7q5+6ah+NtWFPw/70mXXbi6xk51nGXLY9mQghKJlj4/A+JaBSdc5D2TybIQw0QoxAzsBgEpBSIn/3Mzh3Qm0QAu0f/hmRXzi1J2Yw4XR2+Nm7IxzEmc2wZkMCaZlD336F2QwrrsO04jpkYx1yxxace3fxvQV/g9OshEcy3Bf59DM/RLyZg77xdsSajRH1HZyfGceCzFhOtLrw6/DMqQ7+bkV2xI5/JaY0G3fsIBw9EHodKRNwlb0TxMVrA7J3fr9UvXcjzN719uj09ug01g7cYaaVVg6HSE5F3Ps3yDvuR+58GfnK09AZFKfwuJCvPI3c9hxi9QbEbfciLpNRnSqsNo3SOTZKZlvp7Aiajdd4CfQzG2+o9dFQ6yM2XqOwxEpBydT3M12tpMWaefiWIh57vSG0wLW7pod2p4+vbLSTHBO5qfRf/vKXQdu6O8Nm3UOJ6IDKatmLrOQWWqZlWWJegYUTh0XIo7Gp3mdYEYwQQ7XSIGqYyWOjb30a+edfhl6L+z+iGvWjnJk8JpNBR5uffTsdoUm5xSJYszF+QKbmSkgp+beddbxRpzIPFt3Hwwd/SpmjPrxTbJzqEdq4GZEXmcWBt+odfGt7HaAEVf77XbNIsk18b8XpNhdfCCpVagJ+9I6SSQnkZEcr+p//Bw72Exuwl2D6lx+O+ljjvW4GZO+6AvR0Bujq670b4pEtNLj9nuRpsfo+EUi/D7l/J/LFJ5R9waUsXom2+V6YHVa6jLZ7m98naahVvXRD9jkJyMoxU1iqMi0zOVsRbWPTR0CX/O/BFp47fTG0LSfBwtc22SN+j3I5deqrvdRVewdkbfsTn6BhL7aSX2ghfhK9CidqfE4ddXH2hMp6pmWauO7GxIj/jOmKoVppYDCFyGMHkBW/Cr0W625A3HrPFJ6RwWTQ3qqCuL5VdotVsHZjPClpo7vtPnmgNpPZAAAgAElEQVSyIxTEAXx8jpVZgXnIN1vBq/yOcDmR255DbnsO5ixCbLodsXwtwjz2csgVefEUpdio7vTg9kteOHOR9y6emFLO/vzp7YFKlRMdxEm/D/nKM8jn/hTuiwOIjUf74Mcn9GdfjuGyd45LlDNdLklRmfWqDeIAhNmCuPYm5NoblNLlS0/AmePhHfqULkvnqgW0ZWum7mQvg9kiKCy1UVhqo7szQE2lh7pqX9hsXEJLo5+WRj+2mKDZeIl1UifwVzsmTfAPK7PJSbDwywMtSKDJ4eOhl6v50gY7i7LjxnV8n1fSWOelrtpHe8sQdZMoo/k+s+6UtJkljlM8y8a5kx6khI7WAF0X/YYVwQgYUUauvLzchDKiW1lRUbE5uG0rcK7fbl+sqKjoLC8vXwo8DDgAJ/BARUWF73Lbh/mxRkbuKmMmjo1srEP/7j+HDW7L5qF97ttRo652JWbimEwGbc0+9u/qJRBcWLfaBOs2JYxaLexIUy/f2FYb8nW7Y04KH1uVA0BajJW25/6C3P4iNNcP/nBSCuL6WxEbbkOkj61XaMeFLn6wuxGARJuJ/3lXGTET2BMy2dk4efII+h9/MSiLI9bdiLj/I2M2nzaum6lHnj+lArrD+xikBJGdT+J9H6J30cqovhcHAqrErOa8l7bLTOzTs8wUlljJLbBgMs2MSf10uH721fXw/dcb8ASCJfOaUhDeVJI8quPoAUlLk5+6ai/N9WERnP5oJsjJt2AvspKZY57ybOxEjs+BPb00BBUsC0qsLFs9vuB4phCJjNydwPPA2v4bKyoqHhxi34eBD1VUVHSUl5f/PfAR4L+H2W5gMCORvT3oP/5WOIhLy0D7+JeieuJgMH5aGn28+UYvejCIs8WoIC4xeXRBXIvDx7+93hAK4uZnxvLRa8J9alpCEtrNdyNvugtOva3EUQ7vJTQT6O5EvlCB3PJXWLISbePtsHA5Qht5IHZ9URK/O9JGS6+PHk+AV851cue8tFH9HqNhsrJxsrMdWfG/yDd3DXwjvwjt/Q8i5iyckJ9rMHmIsnmY/t+XkU11yJefQu7ZRkgdormenp8+AsmpiJvuVCXJcdGjdNmHySTIL7SSX2il16F66WovePG4w4Fpe4uf9hY/xw4J7EUWCkttUScvPxNZY0/k4VuK+Pb2Wi66A/h1+PfdjTT1+HjP4vRhM2VSSi62h826Q1nXS8jIVqIlOXbLVeM1WDLbFgrk6msMK4KRMKJArqKi4imA8vLy/pt7ysvLvw4UAm9UVFT8qry8PAbwV1RUdAT3eQr4z/Ly8t8OtR0jkDOYoUi/H/2/HoUWlc3AakP7xFcQydHlP2MQWZobfLz1Rm8oloqJFay7IYGEUZY/efw6j+yqo8ejosHUWDNfWJ+PZYgVdyEEzF+Kaf5S5MX2oIXBS9AZvN1KHY7sRz+yHzJzVIbuulsQiVdWSzVpgnfNT+MXbzUD8PTJDm6fk4p5AlaET7e5ONioSkg1AeUT4BsnAwFVgvrMH8DtCr8RE4u46/1Kvt5kTIJnEiLHjvibTyqly23Pqgx2n9Jl10XkE79BvvAXxIbNSukydXIVUkdKfIKJ+UtimbsohpZGPzWVHpob/aGeSZ9XcuGslwtnvaSkBc3GC62Yr5IAYCqYlR7Do7cV863ttdR0qTL3Px5to8nh5RNrcgfdrx09Knirq/Lh7B267y0pxYS9yEJ+kZWY2KsvgBlgRRCAmkovs+cbVgTDMebi04qKinsAysvLBfCT8vLyC8AZoLPfbh1AWvC/obYbGMxIZMUv4eSR0Gvto/+IKCybwjMymGga67wc2OMMeXrHxqkgLj5hdIGBlJKfv9kUkro2a/DQ+jzSYq98uxap6Yi73od8R7kK3nZsgROHwzu0NiEf/zXy6d8jVlyH2HQHlM0bdvX45rJk/ny0jS5PgFann11V3dxQOrryoZHwx37ZuA1FSdiTIpuNk2dPoP/+Z1BfPWC7WL0B8e6/RaRE5wTeIDKIlDTEvR9G3v5u5M6XENueRe8IfufcLuTLTyJffVYpwN52T8REgyKNpgly8i3k5FtwOXXq+szG+wUGnR0BOjtcHO9vNp4+s/qpooWsBAuP3FrEo7vqOdykqm9eu9BNq9PPl9bnY5GChqBZd2fHZcy64wT2IhV4X+3ZVCEEJbNtHN7fz4pgrmFFMBzj7iKsqKiQ5eXlzwNLgT1A/5RDGipoa7/M9gGUl5c/ADwQPC4ZGZFvrDebzRNyXIPxM1PGxvnik/S89nzodfx7/56E2+6awjMaOzNlTCaayrM9HNjdGWrFSUwys/nufBKSRi828viRBrZVdodef3ZjGevnDzbkvuLY3Hon3Hon/oZaXC89iWvb80hHj3rP70fu24HctwNz8Sxib7uHmI23osXGD3mo91zj4Rd7VAD09Jku7ltVGlEz3GON3Rzql417YMMsMlIj0xsR6OzA8euf4N6+ZcB2U34RSQ98DuuSlRH5Of0xrpso5wP/gOn9f4/jtRfoffL3BOqq1PaAH7n7VeTuV7Guup74d30A64KlU3qqV6KgENaulzTWuThzspvq846w2bgfai+ocsyUNCtzFiRRNjeRmJjoDham2/WTAfzw/kwee+08zx5vxgQ4W3Qqnu0gS7cMadZttWoUz0qgbE4i2Xkx0yrInujxSU2VnDpahdsVwO2UOHtiKS6LvtLnaGFU9gPl5eVbKyoqbh5i+/eAZyoqKt4oLy9/EXh/v144U0VFxX9dbvswP84QO7nKmAljI08fRf/3r9OnciFWXo944PPT6ibdn5kwJhNNXZWXQ/udoRKn+ESNdZsSxuT5dLLFyVe21hDsn+em0mQ+tTZnyO/PaMdGej3It15XJsoXzgzeISYWsXYTYuPtCHvxgLccngB/99R53H41Q/zKxnxW2yMnDf2NbbWhQG5TcRL/eF3euI8p9QByx4vIJ38XLqUDsNoQ73wv4pa7xqXqORzGdRP99I2R1HWlavniE2Gfz/6UzVPWBUtWj6q/dKrwelSWrrrSi6N7cPmepkGOPWg2nhWdZuPT8fqRuqS12cfuIw4CnWAVg78rQoPsXAv5RRZl1j1Eqfz3v//9Qds+97nPTcg5j5XJGB/DimAgkbQf8Pb9o7y8/AdAPBAD7KuoqHgj+NYXgF+Wl5f3AB7gk1fYbmAwI5CtTeg/eyQUxFFYhvjIZ6LyQWkQGWoqPRx5M9xrlZCkgrix9Da0O318b1d9KIgrS4vhwdXZEfv+CKsNce1NcO1NyOrzyB1bkPt2hCX33S7k9i0q0Ju1QFkYXHMtwmIhwWZi8+wUnjqpCin+eryDVfkJETm3U62uAdm4d0egN06eP4X+h59DTeXAN665Fu09f4dIG5uKp8HMQ2gaLF2Naelq5LmTYaXLPs6fQv/Jw5CTj7j1HsTaGxCWiVkAiARWm0bp3BhK5tjobA9QXemlocYbeizpOjTU+Gio8REXr1FQaqWg2DAbHwtSKq/Huiof9TVKhMaExqXxWbP0Ulpm4/oliVivINzxgx/8YNC2aAvkJoOiskutCAIkp0Z3JnmqMAzBDaKG6Tw20uVE/+7nwzLmyaloX/4+Im36lIcMxXQek4mm6pyHowfCQVxisgribDGjnxD5ApKvbq3hVJs6XpLNxA9uLyYz/vITxkiMjXQ6kHteU8FbU93gHRKTEdffjNiwmY64NB54uhJ/UEbz4VsKWZg1/vLHf9lWy+EIZeOko1uJV+x6eeAbWblo7/sYYtE14znVEWNcN9HPcGMkG+tUz9ye1wgZQfaRnIa4+U4ljhI3dClytOHzSRpqVC/dkH1aArJzzRSW2sjKndny9pHA2Rs26x4q6wlgiYMDrl5O+Jw4CCCADy/P5F3z04ZdAMvPzx+0rb5+CHuZKWSyxufA7l4aapWCZWGJlaVXsRWBYQhuYDCBSD2A/t+PhYM4swXt/3152gdxBpen8oyH44fCQVxyqom1G+OvuNp6OX55oDkUxGkCPn993rBBXKQQcQmIm+5E3vhOOHNMZeQO7QlnlXu6kFseR774BCmLVnDDrHfxykVln/H48fZxB3InW52hIE4pVY7tmpG6jnz9FeQTv4HenvAbFivijncr8QrD9sNghIhcO+LDn0Le/X7k1meRO18M28h0dSjBoOcrlG3BzXdFvVCOxSIoKrNRVNbPbLzKh88XNhtvbvDT3NDPbLzUOmqhppmM16vTWKtESzpahxYtscUI8gqt2IssJKeaWOqI45uv1eLoCSCB/zvUSmOPj4+tysZkiHdckZI5tlAgVxe0IhjrM3YmYwRyBgbjRD7xGzj6Vui1+PAnEaVzp/CMDCaSc6fcnDziDr1OSVNBnMU6tgfM1vOdbDkbFvX98PJMluRM7kq/EALmLkbMXYzsuhi2MOhT9ZMSjr7F3eeq2Lr680ghONDQS9VFN8WpY5eG/tPR9tC/NxQnkZ80+mBLVp9D//3PB/f9LV2N9p6/R2TmjPn8DK5uREo64v6PIO94N3LXS8hXnoGuoE6b24V86Unk1mdVb+lt9yByC6b2hEdAUoqJRdfEMX+JpLHeR02ll/Z+ZuMet+TcSQ/nTnrIyDJTWKp8zGaK2fhoCAQkLY0+6qp9tDQMbdZtMqmeQ3uRlYzsgdnM3EQr37utmO/uqONEq1qoe+lcJ629Pj6/Po84ixEoD8dQVgSzDCuCQRiBnIHBONB3v4p86cnQa3H7fWhrb5jCMzKYSM4cd3P6WDiIS003sWZjwpjNWs+2u/j5/ubQ6+uLErl7Ag23R4JITkW88z3I2+9XIhDbX4DjhwDIc7Wxpu0oezOXAPD4M6/zT+tyYPbCUffLDcrGLRpdNk72OpBP/Q65YwsDZOHSs9De9wBi6epRHc/A4HKIuHjEbfcib7wTuW+7uuf3lSIH/Mg3tiLf2KoWDzbfh5g1f2pPeASYzEry3l5kxdETCKlb9jcbb2vx09bix2K9eszGpZR0tAWoq/LSWNsva9kfAZl9Zt35lmG9+pJsJr55UwH/ubeJnVVKjfhgYy9fermGr91gJyNuYOXFP/3TP0X095nOXGpFcOGch1LDimAQRo+cQdQw3cZGnj+F/tiXwR9czVy6WpVUTgNls5Ey3cZkopBScvqYO6SiBZCeaWL1+oQxG+52uf3805Yq2pzq+1OUbOPRzUXEmEf2/ZnMsZEtjcidLyLf2MpZkcxDKz4NgCYD/GTfo2SnJSpxlLU3IGJHVm45oDeuJIl/vHZkvXFSSuSebci//h/0dIXfMJsRm+9DbL4fYYusB91oMa6b6Gc8YyR1Hd7er5Quz58avMOs+Wi33QtLVk2r54GuS5obVJaupSlsNt6f1HRlNp5XMHFm41Nx/fR0q+CtvtqLyzn0vDg5VZl15xWO3qxbSskf3m6j4li4CiEt1szXNtkpTZteWabJHJ9AQLL12W68HjUmK6+LI9d+9ZXJD9cjZ/rGN74xqSczCr7R09Nz5b1GSVxcHE6nM+LHNRg/02lsZHsr+ve/Cu5gn1R+Edqnvz7j+nCm05hMFFJKTr3t5tzJcBCXkW0eVxAX0CUP76jnQqc6ZrxV41s3F5I6AtPvPiZzbER8ImLBcsRNd5KelsjJxh6azYlIoaELjRXV++HoAeS256C9BVIzEMmplz3eyVYnvzuiJgKagC9cn0+i7cor/bLuAvrPvwfbngurbQIsXI72qa+jrbgOYZ76QhPjuol+xjNGQghEjh3t+lsQC5YiHd3Q3E+QoqMN+eYu5IE3wGqD3AKEKfozWUIIEpNM2IusFJZYsVgFzl4df7+slNslaW7wc+GsB2evji1GEBMrIqrOPFnXj9ulU13p4dhBF6ePuuloC+D3DdwnNl6jZLaVJSvjmLMghtQM85ju+0IIluTEkxlv5kC9Awm4/Do7qrooSY0hbwxl5VPFZN7fNE3g98lQX6LHrVNQMrULdVNBYmIiwL8O9Z4RyBlEDdNlbKTHrbziWpvUhoQktM99e9iJ63RluozJRCGl5PhhN5Wnw0FDVq6ZVdfHYzaPfeLy60Ot7AiW2QjgofX5zM2IHdUxpmJshMmEsJeQlpvF9gvq/GsScrm5YR8xuk8p/FWfV9m7E4fAZFay7ZdMYn+8t5Emh5oxbSpJ4pZZKcP+XOlyIp/4NfLXP1KBYh+pGWgf+TTing8hEpIi+8uOg6v9upkORGqMRFom2uoNiJXXgc8DDbUgg81Ujm44vE+VXUod8oui2rqgPxaLID3LTMlsK2kZZnQdHA49lKWTOnRdDFBT6aWxTvWPxSdomMZxX+xjIq8fv09SX+vj5BEXRw+6aG3yDygnBbBYBQXFVhYuj2Xhshgysy3YIiSyUZoWw7zMWPbXOfDpEr8Or1d3k2QzMTt9dM+AqWKy72/xiSYunFXPYJdTkmu3jEkdejpjBHL9MB6w0ct0GBup6+j/8304fVRtMJlUJq6wdGpPbIKYDmMyUUgpOXbQRdXZkH0m2flmVl4bP67G/11V3fzvwXAw8r4lGdx6hUBmKKZybLITLLxZ38tFl5+AMGFZsJTFjpqBpY4X2+DQXtXD1tMNmTmI+EROtjj53dsjy8ZJKZH7dyJ//B04eTjcC2cyIW67F+3BhxAFJVHn1Xg1XzfThUiPkUhMRixbi7juZuX8XF8dLrt3u+DEYeT2F8DZC3mFiJjpMWkXQhCfYCKvwEpRmRVbrMDl1EOlbgBej6S1yc+FMx4c3QEsVkFsvDbm6zLSY6Pr6vxOH3dz5E0njbU+nI6ByiWaBjn5FuYtjmHJSlW+FzeO32E4chKsrLIncKDeQa9PRwIHGnpx+QIszY2PuvvZpUz2/c1iEfR0BegJWj1IqcbqasII5PphPGCjl+kwNvLZP8KOLaHX4kOfQLtm3RSe0cQyHcZkIpBS8vZbLqrPh4O4XLuFFeMM4qouuvnOjrqQ6fdqewIPrhqb6fdUjo0QggSbxhs16h5dTTx3/MP7sC5cBn4fNNWHsxJeL5w/hdz2HLLyFD/uyaXZpwK34bJxsrEW/b8ehVeeAk/Y6oG5i9E+9TWVBYmCMsqhuFqvm+nERI2RiI1DLFyO2HQ7xCVAQw14ggJJfh+cO4l87TnoaIXsvKjKJF8Js1mQlmGmeJaVzBwLSOjtCYTWV6SEni6duiql9BjwS+ITtFGXIkZibKSUdF0McO6km8P7XdRUeunp0rlUFiI908TsBTEsWx1HQYmNxCTTpIhpJMeYWV+UxPEWJx0uFfCfbnNT3elhdX4C5igW9JiK+1tMjEbtBfU87ukOUFxmjUj2d7owXCAXnU9BA4MoRH/zdeSzfwq9Fjfdibb+1ik8I4OJQOqSw286qasKN0vkF1pYtiZuXA94hyfAd3fW4wlGcXmJVj67LhctyldfL8daeyJ5iRYaenz0enVePtfFPQsWIuYsRL7nIvL1rcidLw0ohTxZ28GRTLWSqiF5d9HgVVXpdiGf+zNy69NhPztQRszv/lvE6g1Rv2JtYCDiEhC334e8+U7k3qDSZV8fnd+vLD5efwWWrkHbfC+ibN7UnvAoEEIFdGkZZhYuj6U+aDbedTF8vTodOqeOKpXfrDwzRaU2MnMm3mzc6QhQV6383np7hjbrTkjSsBdbyS9UWbepIjXWzHduLuT7bzSwr84BwN5aB1/ZWsNXN9pJGUXP9EwnNcNEUoqJ7k7DiuBSjIzcKJBSTosJxHQ5z0uJ5lVsWX0O+ZNvhyeWC5ajffSz00qRbCxE85hMBLouObTPSX11OIizF1tYvnp8QZwuJd/bVc/ZdrUyH2PW+ObNBeMy/Z7qsdGEwGrSeLNeTUCqu7y8Y04KJk0gbLGI2QsRN74DUTwH6XJCayM/mXs/zbHKPHlT0wFuevJRqKuGhCRIz4KDe9B//G04diBcRqlpiJvuQvv4F9GKZ0+Le9tUj43BlZmsMRImE6KoDLHpDkRhKbK9BS6GlQtpqlNm9qeOIBKSISt3WnzH+zCZBClpZorKbOTkmxECHD2BAZ5rvT069TU+ai948XklcfHasL6box0br0entsrLsUMuThxx097ix+cdmHqzxShT9MUrYpm7KIb0TAsW69T/nc2a4NrCRNx+ndNt6vnQ4fKzu6aHZbnxJMdEXzA3Jf3ZQmAyQVO9yl729gQonm2bVtfKeDAychFC/9Zn1UQ+166MP3PsiLxCVR5hjR4VnV8ebGFfrQN7kpX8ZCv2JCv2JBv2JCvJMaar5osfKWRnB/qPv6NKxACy89E+9vlpoUJmMHL0gOTgXieNdeEgrrDUypKVseO+Zv74dhsHGnpDrz+zLofC5Oi5Z4yVG0qS+MPbbVx0+bno8rP9QveAUkmhmWDpKkxLV3H8TB1vv6mCPk0GuL/6VQgEkAfeUOp+ickDe+wAZi1A+8DHEPaSyfy1DAwijtA0WL4WbdkaOHsC/aUn4O03wzucPYF+9oTqn7vtHpV5Nk+vPqDkVDOLV5hZsDSWxjof1ZWekNogKMXLsyc8nD3hISM7aDaePzaz8UBAWSXUVXtpafSHKrn7YzKrknh7kZWMLDMiSssVTZrgoyuyyU6w8oOKl9AldAMPnjjA+5dk8MF33jjVpxgV5BVaOXHEjdcjcTnV+F+NVgSXYgRyI0T6/arWPRCAhpqQvYoEEAIysoOBXYGSGs6xq//HxU/6udZ2eWnp9dHS6+NgY++A9+KtmgrwgoFdX7CXk2CN6prsqUL6vOg/fRg6gyuocfFon/wqIi5hak/MIKIEApIDu3tpbvCHthXPsrLomvEHcfvqegZ4B927II1rC6dPX8xwWEwad89L5f8OtQLwxIkObixNxjTEveTPteGZ1sYkL7n52XCuX2aifxCXmIy4/28R624wFp4MZhRCCJizENOchcj6GuRLTyD37whXezTUIH/1Q+STv0Pcchdiw22ImJF5M0YLJrPAXmzFXqzMxmsqldl4f4GUtmY/bc3+kEJkYamVxOThF0ellLS3+qmv8tFQ5x1kFQBqOpaZY8ZebCU7zzIudeHJ5h1zU3ng558bsO0IkLvzODeVjV4Qa6ZhMgkKS60hK6ALZ72XDeR6vQEsJlU1MtMxArmR0t4ysF+jP1IqKfrWJuTRt9SmvvdS0sKBXV6ByuTl2iExZcImKI093su+1+tV6fu+FH4fZk0pKdmTVfYuvy/IS7ISb706M09SSuRvfgwXzqgNmob2sS8gcvKn9sQMIkrAL3nzjV5am8JBXOkcGwuWxYz7Gq3r9vAfuxtDr5flxPHBpZnjOma0cdvsFP5yrJ1en05Dj5d9dT2DAtUTLU6ONKlSHE3Ae25YiOmuR5B1VcgdLyL3vKYETYSG2LQZ8a4PGoslBjMekV+I+Ohnke/6IPLVZ5A7XgoL+3S2I//yK+RzFYhNtyNuunNaWtwkJJpYsDSWeYtiaG4Mmo03hu+1Pq+k8oyHyjOekNl4cvLA9Fp3Z4D6ai911V7crqHNulPSTNiLreQVTF9penmpEkuQ/9zbRGNrF+9fXYA2w9s5rkTxLBvnT3mQEtpb/HR3BkhKCc9RPX6d3x1p5bnTF4m3mvjM2lxW2Wf2s0Rc7osTBciGhoaIH3Q8jvTS5VT17I210FCLbKqDxlpobWbIvP5wxCWEA7ucYKlmXoEy0h3nheoL6DT2+Kjr9lDX7aW+y0tdt/rP7R/leaIacu39Ajt7ssrmpceZIyrUMJ6xmQj0LY8jn/h16LV47wNoN71zCs9o8om2MYk0fr/kzdd7aWsOTyxmzbcxb/H4gzinL8DnX6ymrlstrGTFW/j+7cUkjcD4eiRE09j89nArfz2usmuz0mJ4bHPRgL/f116t4e1gIHdjaTKfWZc74PPS7YQTR9SiV6598k58goimsTEYmmgcI+l0ILdvQb76LHR3DnzTbEasuxFx812qpWMa4+zVqavyUlPpweUcPAe1WAS5BRbiEzTqa7x0dw49b4lL0LAXWcgvspKQOP0XnPXn/kTBxz43aPvKR18FYH3HcT5lqcQyaw6ibD4UlExJ+e1UXztv7e6lsValYwtLrSxdpTLWx1uc/GhvI409A1O1712cznsWZ0xbYTGAvLw8ULazgzAycqNAxMZByRxEyZwB26XPC831yMZgYNcYDPaa68M+MpfidCgZ4nMn1TH6tlttwcAuWJqZq0o1ycwZsdS2xaRRmGKjMGVgD46Ukg6XXwV1XV7qg4FeXbeXdudlzhNC/S9Hmwc2t9pMIpi5s5GfbKUgGOjlJVmnfTpbHt6HfPI3oddiw22IG98xhWdkEGn8Psm+XY4BPRxzFtqYs3D8QZyUkv/c0xQK4qwmwZc25EcsiIs27pybyjOnOvAGJOc63Lzd7GRpjiorP97iDAVxmoDyRemDPi9i4mAG23gYGIwEEZeAuOPdyFvuRu55TSldtgQXtPuULne9rARRFq9ELLoG5iyKqh79kRAXrzFnYQyz59tobfZTU+mlqd4X0jfy+SQ1lUNXFlmsgvxC1feWkj5zev71bc8hn/7DsPvsSltIe2csDz3+GxL9TrBYoXgWomy+Uj0tm4dITJ6kM546SmbbQoFcXbWX0oU2/nSijedPX2So1NSfjrZzvsPNZ6/NI2EGVpgZGbkJRAYC0NYMjbUqsGusDQZ7dQN9kUaCyQxZueEV674gLyc/Ijdxpy9AQ7fK4tUHg7v6Li/1PV78+ui+IwJlGNxXnmlPDpdqJtkuf+Od6lWePmRdFfojD4XHaM5CtH/85rRrPI8E0TImkcbnlezb6eBieziIm7c4htkLIiNn/Pjxdn5zuDX0+rPrcrmhNLIP2Ggbm5/vb2LLWZVFWJoTxzdvUlmDK2XjZiLRNjYGg5kOYyT1ABzeh/7iE+ES/0uxWmHuEsTiFYhFKxCZOZN7khHC41ZZuurKwbYBmgly8izYi61kZpvRxuHlGY3oe7cjf/kDAMr3nlYVW7kF4HEhXVYxnykAACAASURBVE5u+sAXeDludmj/PGcrXzn6v+S62gcfLCsPMWu+CurK5itxvgiXY071tSOlZOfLDro71fP7hLmX3e6wyn28ReODyzLZU9sTevYA5CVa+NIG+6Akx3RguIycEchNAVJKuNgWztyFAr06cHSP7mBCKNnu3HD/Xej/EegxCeiSll5fMLjzUNvlVf/u8tDjHX2ZZqJVU0IrydZwoJdkIzvBQnZW5tSPTU8X+nc+F/a+Ss9C+8oPEIkzQ5xitETD9RJpvF6dfTt66ewIB3ELlsZQNi8yQdzhxl7+9bVa+tY/3jE3lQdWZkfk2P2JtrFpdnh58JnK0O/92OYivAHJl1+pAVQ27qd3lpKbOPNVxqJtbAwGM53GSEoJZ4+jb30Gjh8Cr+fyO+fkq4Bu8QqYvQhhmV4LkFJKdF8Cx4604vdJMnPM5NqtUWEVMBHII/uVoFqfX0PZPLVwbAs/j6SUPHminV8fDn9fk/xOvvj2r5jXXT38D4iNh7K5iL7ArmT2uMVzouHaOX/WzYmDSuuhRwaoCLQigRV58fy/NTlkxFkI6JLfHm7lyZMdoc/FmAWfXpfLddNMcMwI5PoRDV/A4ZA9Xf0yd+H/c3EM55ycFgzs7JBbGMrkkRQZoZVutz9UmtkX3NV1K8XMUSbxMGtgT4kjN14LK2oGg704y+SkwqXfh/7vX4czx9UGWyzalx5F5BdNys+PRqL9ehktHo/O3u29oZU8gEXLYymZE5kVumaHl89tqQotcizIjOVbNxdOiCJsNI7N999oYGeVWoy6tjARhzdw1WXjIDrHxmAg03WMpM8LZ48jjx5EHnsLmuovv7PVBvOXIhZdo4K7jMgvKE0E03VsRos8fQz9h98AX7CMNL8I7fPfRcQPvQj/Rk03/7G7EW9ATbAsGnwm18F1rUeR509C9bnLt/P0ITSwF6mgrmyeKsnMyB7VnHCqx+dIUy8/3dPETZ5UYoXKNu4Undy+MpUbSpIG/S67qrr50d5GPIHwxPTeBWl8cGnmkArL0YgRyPVjqr+AY0W6ndBYPziD19o0BqGV+CEyeAWQlhmRFLy3n9hKf6GV+m4Pbv/ov2/pseYBfnj5wSAvPdYcsfp4KSXytz9R/QcAQqB94iuIpasjcvzpynS9XobC49bZs91BT1f4elmyMpaissgEcR6/zkMvV3PholotT4s184Pbi0mNnZhW5Ggcm6qLbj7zQtWg7VdTNg6ic2wMBjJTxki2NiGPHUAePQCn3w77nQ5FboEK6havhFkLojZbN1PGZjhk9Xn0x74M7mALR2YO2hceQaSkDfu5020uvrO9ji5PeDHyQ8syuW9BGvh9UH0eef6UCuzOnxosmjMUyamhoE6UzYfCsmG/G1M1Pk5fgP872MpL59TvtFJLYJmmgt6kdI2NN18+y1Z10c13d9bT5AgLoSzLieNz10+P3nUjkOvHTLtBKKGVhn5CK8Egr7mBIU1WhsNqU2UZwcAulMHLzB2x0Mqw5yol7S5/UGjFO0BVs911hVWkIYgxa/3KM/vMz23kJVqwjFJsRX/1WeSf/jv0Wtz3YbTN9436nGYaM+V6cbtUEOfoDgdxS1fFUlgamSBOSsl/7G5kezAbZdbgOzcXMS8zNiLHH4poHZtvvlY7wPwc4KbSZD59lWTjIHrHxiDMTBwj6fXAmePhwK5lmDmULSaYrQv21qVHjy3KTByb/sjGOvRHvxhupUlOQ3vokRH3Nzb1ePnm9jrqu8NB+y1lyTy4OmdA9YeUEtqaQ0GdPHcK6quvvPhvNkNRPxGVWfMQSWHri6kYn0ONvfx4byNt/YT5sixm7pIZodcbb0scYEVwKQ5PgB/sbhjwfMqKt/ClDfmUpkWmtWKiMAK5fsz0G0QfUg8KrTT0K9NsqoOG2jEIrZggK09l73IKgrYJdsi2I2yRmQg7fQF6RRzHalpC2bu6bi+NPV5G65igCXVx2vtZJahAzzbkyos8fgj9h/8aurmJtTcgPvrZGaOGNR5mwvXicursec1BryP4RRKwfHUc9uLIZYeeO93Bf7/VEnr98dXZbJ49sZ5P0To2x1ucob44uPqycRC9Y2MQ5moYI9nSEC7BPH0sXMI3FHmFIcEUZs2fUnGvmTw2sr0V/dGHoCP4+8UloH3hu6Nu4ejxBHhkZx3HWsLzuWU5cXxhff6w3r/S5YQLZ8JZu8rT4HJedv8QmTmhcszUlevojEtEaBOfyer1Bvjfgy1sPd81YPvaggQeXJXD+YMeGusGWxFcDl1K/vh2GxXHwkIxVpPgE2ty2FQSvYqfRiDXj5l8gxgJSmilHZqCAV5DLbIpWKbZ03XlA/RHCEjLVA+AXLuyTcgrHLPQylBjE9AlzY5+nnhB64S6bg+OMYitJNlM/fzwrOQHesj79aNkdtZjQkLJHLTPP4ywXD2TzuGY7teLszfAntd6cfYGg3QB16yNI68wcuN7vMXJ17bW0Fd+f3NZMp9ckzPhCwHROjZSSr74cg2n2tQE42rLxkH0jo1BmKttjKTHA2eOIo8eQB47oNoyLkdMrMrWLV6psnWpgy1DJpKZOjayp0tl4vr6Gm0xStikbN6YjucLSH68r5HtF8IieUXJNr52g53M+JEF4lIPKOG98yeVJdb5U9DSeOUPxsRCaX8RlTmIuPgx/R6X40C9g5/sbxpgj5VoM/GxldlcX5SIEIL2Vj+7tzkApW56y51JWG1XrsjaV9vDv+9uxNUvU3Dn3FQ+ck3WhPS0jxcjkOvHTL1BRALZ0x3M3AW98Bpqoak2vHI0GpJTg4FdsEwzJ1immZwaEfsBKSXdnsAgoZX6bi/NDt+QXiLDYdF95Ho6sZfasacnhP3xkqzEWqa3J954mM7XS68jwJ7XHCHDWaHBinVx5NojF8S1O33845YqutyqX2F2egwP31I4KT6K0Tw2lR1uvr29jgSbiX+5wU56XHT24kwU0Tw2BoqreYyklKolo68E88yx4Vsx7MVhJczSeRFptRiOmTg20uVEf+wrUHNebTCb0T71dcSCZeM7rpT8+Wg7fzwa/nulxpj46qYCZqWPrVxQdndCpSrFlOdPQdXZK7fqCKEW9ftEVGbNR2SNbQHP4Qnwy4PNbKscqOJ+XWEiD6zKJiUm/P1TVgQ9IdP40ShQ13V5+O7O+pDf6/9v796jo6ruBY5/z5mZhAQSIIRXeMtLHioGCohatOJFrYr22u2rtbX22lpdWrW1D2+1xUdrW6l1eW219trW2mu3ttW+VkUFH1UQERUUkIcIhEeUVxLymkzOvn+ck8xMmAmZZCZzJvl91poVZufMmXPyY87Mb/bevw0wbUgBt5wyggEZmtveWZLIxeiJF4hMMw31sDea2LWuhffRns4XWvESO6tsFAwbCYOGMHjIkLTEpjHisKcmpsiK14O3qzocV7Woo0oLg61DM1uHaRbnUZLGYit+5efXizGG5oi7eGykydDk3SJh9+fm9Q001Lvxtm2YdXJfhpZ1LKFwjCHcbGhocmiItNxMzL/d23NbqthywC2B3D8/wL1nj+3wN6Fd5efY9HYSG/+TGEWZxgbYuM5L7FZHl99JpKAQpszwhmGWYw1If29dT4uNCTe60zc2ves2WDb2V2/BKp+XtudY/kEVD7y+p3UqSn7A4hunlDF7ZFGX920iTbDjg9bhmPYHm3COVkl96gwCNy5O+blWVdTw4KpKDsbUTeifH+Ars4cmXTJgxweNvPOGOwKkoK/NGecUYXWwV62uqZmfr9jDyp2HW9sGFQT53ukjGTfQP/PmJJGL0dMuENlkmprcydRxyyXsdIcNpFxoJY/giDE0Dy6LVtIsS1+hFXA/nO+vc5dM2Ln8RSoqKtlVOIRdhUM4mJ/6miIFQfuI9fBG9M9jeL88Qj1kwdJMvl6amw1N4TZJWFN8W+vvWtuI+12HLl+WITLGob7AoTFiqG9yaIw41EdafroJWmxbqtVVbQsWnzGK44amd2hJe+Ra5l8SG/+TGCVmjIG9uzDrVrtDMDe9B83tFCMbNc7rrZvlDrULdH3eVE+KjYlEcH75I3hnVWubdcV12Kf+R8LtR4wYcUTbrl3tLDERY11lLT98eRe13rQT24KrZg7h3MntV8JM1aBBg9i3aQNmi1dEZetGqNgWXQsPsM67BPv8yzq8z5rGZh5ZXdlaMKzFqWOKuHrWUIr7JP8c2BwxPPe3aprC7vv27E/2Zejwjn+h6hjDn97bz+Pv7MPgJnJLzh7rq145SeRi9KQLhF+5hVY+ilsmoWXZhNZSux0VCMDg4XHLJFjDR7nVNfM7922J8/K/MI892HrfOu8S6s+62KukGVNRs8ottpJqJ55twbB+oSPWwxtZnE9RDpS5jZXs9eI47SRe4fiEy03EHMLh6DbNTXQsCeuiJuOw1DnEHtPOJP80+FL5EBZNSe+b5dHItcy/JDb+JzHqGNNQDxvXRufWHfg4+caFfbGmngjTvd66/p0r+NRTYmMcB/PofZiVL7a2WRddib3wwqSP6UoiB+5wwcUvVlAZU2b/3MkD+VL5kLStmZYoPqahHj7c7PXabcReeCHW5OM6tL+VO2v4xaq9HGqILqkwoE+Ar84exkmjOtajuGFtPTVVzYybmE/p0M6Nllqz+zD3r9jDd+aPZHJp5ipOd4YkcjF6ygUiFxlj4NCBmAQvZsHzVAutAAwaErdMQmuil2QxTfAW4PzZ96DZu2DMnId99S1J18+LxBRb2VUVZmdMRc3aThRb6Z8fYGT/mPXwvESvtDDUbQtTGscQibjJVlyPWNihMWyob3RobDQ0hh0gSF1tmOYmdxij0ww0G3D80eMYMYZGHJowhHEIG+8nhnocNjv1VNN89B0lkBewKAja5Adt76dFn5BNn2DLzSI/aDNpUAGnehOvu5Ncy/xLYuN/EqPUGWPcAmnveknd5vXt99aNHh+thHnMpA5XOewJsTHGYJ74FWbZ31vbrLMvwv7MFe0+rquJHMChhgh3v1TB+/saWttmj+zHzSeX0SfY9fnb6YpPdUOEh1dX8sr2mrj208YWc9WsoSmt72aMSct7cGPEIT8Nf6N0k0QuRk+4QPRE5nA1/etqOLTx3fgFz9v79i+Z4gEJFjwfCU1NOHffDIe9i8boY9wFODvRs2eMoaqh2ZuH1xgzFy/Mx7WpF1vJC1iUFUWraY70evPKivPiLrzGuIlXXYNDbUMzdQ0O9Y2GhkaHxrBDuLGl18sdbuBEDE4EaAbLAcuxCBh/JGHNMUlXGEPYtPzbaXM/UZv700BrUhVNsLz7IZs+ATsm+bJikrJoclYQtOMStPygRX7A7rbEurPkWuZfEhv/kxh1nWmogw1rvWGYa6C9eVN9i9zCHsfNwpp2IlbxgKSb9oTYOH/9A+ZvT7Tetz55FtbnrjlqspGORA7chORnr+1hxc5okjS+pA//fdpISro4ZDAd8Xl1RzUPraqMW9h8YEGQr80empZ5fT2NJHIxesIFoqdK2l2/tyJmDp7Xg/fxnrjx2B1i2dHiLMUDsG+9F6sk/YugNkYcdteEj1z4vDpMOMVxmqfaxQwN5JGHRdBYBLGwE7+Wu5Vj4hOsJi/JasTQlCDpasJgBcAKWthBQzBokeclUPlBm4KQmzwVhNr0gMUlZzEJW8jdJi9g9fiCM8nItcy/JDb+JzFKL2MM7NoerYS5dUN05EtbluX11s3Cml4O4ybG9dblemyc5/+K+eMjrfetT5yK9eWbOtQjma5EDtz36d+99TF/2XCgtW1wYZDvnT6KMQM6vwZwV+JzqCHCw29U8uqO+F64Tx1TzFXlQ+mXY9NPuoskcjFy/QLRk6W0/EBTk1s1c+9Ot5pmS5JXuav9RU8BgiHsb9zV6bVbOssxhn21kdYKmtGqmo0cbEj8hnduoIRhVvrXtEvW+9WEwbENJgDYEMwLgNWMHYJA0CIUssjL95KsvCRJVoKesZDdexOuTJFrmX9JbPxPYpRZpq4WNrwTTeyqDiTfuF8R1tRyOG4m1rRyBo87Jmdj47y2DPPofdGG6eXY196a1QXW/7X5IA+9UYnjfdwvDNl869QRzBjeueJcnXntGGP49/YaHl5dSXVML9yggiBfmzOMWSNSX3u4N5FELoZcvP0rHbGJFlqp8NbD8xK9vRVQXweBINaVN2DPmZ+mo06Pw+HmI9bDq6gOM722kFFW/NDPJuP2cEUs9+bYBscGY+P1erk1YgIhi2CoJfmyyM+z6ZNvUZBvUxAKtA4rjB1u2Lbaprxe/Eti418SG/+TGHUfYwxUfBithLl1Y/IRNZZFaOJUIpOPdythjhmfdA6735i3V+L84kfRc5swBfvri7HyO9/7lS5rdh/mnld20+CtTxCw4JrZwzhzQvIhrsmk+to5WB/hl2/sjSvxD7BgfH+uLB9CvzzphTsaSeRiyMXbvzIZG2OM+41gIIRVlPpSA9ly8GCE/Ycj2EEo7GPTt8CmMC/QbfO35PXiXxIb/5LY+J/EKHtM3WFY/7ZXNGUNVB1MvnFRf6xp5TC93J1b18+f799m41p3rbiWpZdGjsP+5l1Yhf7padp2sIE7llewP2aNtoumDeLyE0qxUxgx09HXjjGGlz6s5pHVldTEFIcrLQxy7ZxhlJf552/jd+0lcv5ZJEGIDLIsCzKwcGmmDRwYZOBAeZkKIYToGazCfjDrFKxZp2AcByq2RZc32Pp+dC47QE0VZuVyWLkcY9lu9cvpM7GOmwmjjvFFb535cDPOA3dFk7ghw7Fv/L6vkjiAcQP78JOzxnDHixVsO9gIwFPv7Wfv4TA3nDScvED6/pYH6iP8YtVeVlXE98ItnDCAL5YPpjAkvXDpIp8QhRBCCCFEt7Ns2y18Mno8fFphamso2rmV6teWu711sUsTGad1AWrzzONQPMDtrTtuJtbUE9tdeihTzJ6dOD//PjR6a+QOKMG+cTFWcefW0Mu0QYUh7j5zND/9927e3F0LwL+317CvNsKt80e0u/B2RxhjWL6tmkferIxbomlI3yDXzhne6Xl5IjlJ5IQQQgghRNZZfYvoc8oCDh87w+2t27E1WjBl2yaInQ5UfQizYhmsWOb21o2fHN9bl+ECW2b/RzhLbosuadS3yE3iSodm9Hm7qjAU4Nb5I3nkzUr+uekQABv31XPL0u3cdtooyoo7V2Btf10T//P63tYEscXZEwdwxYnSC5cpksgJIYQQQghfsWwbxk7EGjsRzr0EU1ONWf8WtMytO1wd3dg4sGUDZssGzNO/h/4lWNNPdAumTJmBVZjeniBTfchN4g7tdxvyC7BvuB2rbHRanydTArbF1bOGMqxfHo+u+QgD7Klp4pZnP+Q780cybUhhh/dljOGFD6r43zc/orYp2gs3tF+I6+YM4/hh0guXSZLICSGEEEIIX7OKirHmzIc5890K1du3RufWfbg5vreu6gDm1Rcwr74Atg0TpkR760aM7VJvnamrxbnvdvjIK8gXDGJf+12scZO6eIbdy7IsFk0pYUi/EEte3U242VATdrjthZ1cP3cY88f1P+o+Pq51e+He2hPfC/fpyQP5/AmDKQhlfw5jTyeJnBBCCCGEyBmWHYBxk9zk6fxLMTVVmPfWwLo17s/amAWnHQc2vYfZ9B7mz7+DAYOwjpvpLkY+ZQZWQQq9T42NOA/cATu3tRwI9n99E2vKCWk5r4suuuiItqeeeiot+07mpFFF3LVgNHe+VEFVQzMRx7DktT1UHm7is9MHJUx6jTE8t9XthauPRHvhhvULcf3c4Uwb2vG/qegaSeSEEEIIIUTOsor6Y809Heae7vbWbduMeXcNZt1q2L4lfuND+zGvLMW8stRddHXCVKzp5e4wzLLRSXvrTCSC89A9sHl99Hm/cB1W+UlpO48VK1akbV+pmFRawE8WjmHx8goqqsMAPL52H3sPN3HN7GFxa8zuqW7gjmU7eWdvXWubBZx37EA+d8Jg8oPSC9edJJETQgghhBA9gmUHYPyxWOOPhUWXYaoPYt715ta99xbUxZTEb26G99dh3l+H+dNvoaTUHYI5fSZMOR6rj9uzZBwH8+h9sG519HnUVdgnL+ju08uYof3yuGfhGO55eRdrK90k7YUPqvi4tolvfXIEhSGbZzcf4rdvb6a+qbn1cWVFeVw/dxhTUphXJ9JHEjkhhBBCCNEjWcUDseZ9CuZ9CtPcDNs2RefW7dgav/GBfZiXn8W8/CwEgjBxqjuvbu8uzKqXo/v8tMI+c1E3n0nm9csLcNvpo3hw1V6WfeAu/bC2so5vL93OgD5B1lVGe+FsC84/toTLji+VXrgs6lAip5QKAD8AZmmtz/LaFgA3ArVAhdb6ps60CyGEEEIIkWlWIOAWPpkwBS78HKbqoFsBc91qzPq3oT6maEdzBDauxWxcG7+P087BWnR5Nx959wkFLK6fO4zhRSEef2cfADurwuysCrduM7I4j+tPGs7k0oJsHabwdLRH7jzgH8BcAKWUBXwHOEdr3aiUulMpdSbwfCrtWuvn0n5GQgghhBBCHIXVfyDWyWfAyWe4vXVbN0bXravYduT2s+djXXp1xtaoe/LJJzOy31RZloWaXsrQviHuX7mXiONWBLUtuGzmSBZNKCQvIL1wftChRE5r/TSAUqqlaRKwXmvd6N1/GvgMsCPFdknkhBBCCCFEVlmBAEyahjVpGnzmCsyh/V7BFHd5A2vqDKzLr3HXt8uQefPmZWzfnTF/XH9K+4b49ZuVFIQCfPHEwZw0eRT79u3L9qEJT2fnyA0CDsTcP+C1pdoeRyl1NXA1gNaa0tLSTh5ecsFgMCP7FV0nsfEfiYl/SWz8S2LjfxIj//JNbEpLYcJkuODSbB9JVs0vhflTowud+yY+Auh8IrcfKIm5X+K1pdoeR2v9MPCwd9dkIuMvLS2VbxJ8SmLjPxIT/5LY+JfExv8kRv4lsfE3iU/3KysrS/q7zvYPbwGmK6XyvfsXAC91ol0IIYQQQgghRIpSTeTCAFrrZmAx8IRS6vdAPrA01fY0nYMQQgghhBBC9CqWMSbbx5CM2b17d9p3Kl3C/iWx8R+JiX9JbPxLYuN/EiP/ktj4m8Sn+3lDKxOWSpXaoUIIIYQQQgiRYzpb7EQIIYQQQgiRJvfee+8RbTfffHMWjkTkCknkhBBCCCGEyLIlS5Yc0SaJnGiPDK0UQgghhBBCiBwjiZwQQgghhBBC5BhJ5IQQQgghhBAix/h6+YFsH4AQQgghhBBCZFnOLT9gZeKmlHozU/uWm8Smp90kJv69SWz8e5PY+P8mMfLvTWLj75vEJ2u3hPycyAkhhBBCCCGESEASOSGEEEIIIYTIMb0xkXs42wcgkpLY+I/ExL8kNv4lsfE/iZF/SWz8TeLjI34udiKEEEIIIYQQIoHe2CMnhBBCCCGEEDktmO0D6Ail1K8ABygBntFa/14ptQC4EagFKrTWN8Vs/3XgC1rrE2PaLgPOB6qARuDbWuu6BM8VAH4AzNJan9Xmd0fst7dLJTZKqR8CpUAh8JbW+qde+wnA3cBhoA64WmvdlOC5vgGcBOQBWmv9mNdeADwARLTWX8nk+eaKbMdFKdUHeAhoAAYCy7TWv8zkOeeKbMfGa98B/NPb7COt9W2ZOt9cku3YKKVOBj4fs9mZwMla670ZOeEc5IMY5QH3e23NwKta699k8JRzRjfHJuH7fnuf4XqzdMTG+91lwL1a6+HtPNflwMVABFiptf6x1y6xyYCcGlqplLKBl4FTgeeBc7TWjUqpO4GXtNbPKaXm4f4HvF5rvcB7XF/gz1rrhd79ycBCrfX9CZ7jAqASuKPl8V77EfsVUR2JTZvtlwIXaq1rlVL/AD6vtT6glPoyYGmtf9Vm+wnAbVrrK7zn+hdwkda6Wil1JbAb+KzW+ssZP9kcks24tNnuBeB8rXVtps4112T5NfO8XMeS88PrRilVDPxaa/3ZTJ5rrspWjIBLgAat9e+87W4HHtRaf5zZM84dmY6N95iE7/vJPsMJVxdjMwE4Hbg42d9WKVUEPAmcrbU2SqnHcGOxSWKTGbk2tDIP2A9MAtZrrRu99qdx/3OhtX5Na/3XNo+LAHlKqZYeyBJgbqIn0Fo/rbVekaA90X5F1FFj00YEqPN6biJa6wNH2f4M4BkArbUDLMWLodb6UeD9NJ1HT5O1uLRQSpV5+63v2qn0ONmMzUCl1N1KqV97X1KJeFl/3QBfRYoKtCdbMarDHWXQYhhQ3rVT6XEyHZuk7/vJPsOJVp2KDYDWekuipLqNecBzWuuWXqJngNO8x0tsMiAnhlbGWAz8GBgEHIhpP+C1JeR923AH8JBS6jDwFpCvlCoE/oi70N7TWutHMnbkPV+HY6OUugH4jfdtTQlwqM32JUqpsbjDJsAdojcI2NTefkVCWYuLUmoOcBMwA7jU+zAkorIWG631TG+/hcDflVIXaq2r0nRePUFWr2fe8L3TW4YkiYSyFaM/AN9VSj2MO/zvEO4QNBGV0dhorf+WqQPvBToVm2Q7a/s5Gjfpa7vfiWk5cpFQziRySqkbccfqvuoNjSyJ+XUJ7jcMSWmtlwHLvH1NA8q1O0fuvAwdcq+RSmyUUgoIaa2117Sf+G83S4ADWusPgXNjHleWYL9r03kePU2246K1fh242LvQ/0Upda5OMNehN8p2bFporeuUUm8CY4F3unhaPYJPYnMFbsIgEshmjLwvpO6M2e4nwPZ0nFdP0B2xEZ3Txdgk1PZztFJqITA92X5F+uXE0Eql1DVAtdb6/7ymLcB0pVS+d/8C4KUO7iuI+43EY2k/0F4oldgopRYBx+qYibNet36e901c3PZtvOj9rmWM9wLgjfSeTc/hs7iEgXxy5HqTaX6KjXInn0/zjqHX80NsvPuXAE+k78x6Dj/EKOZYJuMOq3y762eW+7oxNiJFXY1NCl4HFiilLO/+Itw5eSJDH7CKAQAAAUFJREFUfF/sxJu/8QTu+PQW38X98HE9brWdPcAtsd2/Sql/aq3Pibl/I3AM0Yo97X7L0PbxR2vvjVKJDTAaWAXEDom4V2u9QSl1PG4loxrciqLXxYzbjn2+m4BP4CYFf9JaPx7zu1HAf2upWumLuCilzgIu856rGPit1npp28f2Nj6JzUDcqnstsXlIa93rPyz5ITZe+wXAMVrrJek9w9znhxh5ScaduHOHBgC3a623pfVEc1B3x8Z7zqTv+/JZLSpdsYnZX7t/W6XUpcB/4r5GVrdNCCU26eX7RE4IIYQQQgghRDwZ6iSEEEIIIYQQOUYSOSGEEEIIIYTIMZLICSGEEEIIIUSOkUROCCGEEEIIIXKMJHJCCCGEEEIIkWMkkRNCCCGEEEKIHCOJnBBCCCGEEELkGEnkhBBCCCGEECLH/D+EfsZYyDCtYAAAAABJRU5ErkJggg==)

#### 공휴일 데이터 추가

```python
## 공휴일 데이터 생성
date="""2018-09-23, 2018-09-24, 2018-09-25, 2018-09-26, 2018-10-03, 2018-10-09, 2018-12-25, 2019-01-01, 2019-02-04, 2019-02-05, 2019-02-06, 2019-03-01, 2019-05-05, 2019-05-06, 2019-05-12, 2019-06-06, 2019-07-17, 2019-08-15, 2019-09-12, 2019-09-13, 2019-09-14, 2019-10-03, 2019-10-09, 2019-12-25, 2020-01-01, 2020-01-24, 2020-01-25, 2020-01-26, 2020-01-27, 2020-03-01, 2020-04-15, 2020-04-30, 2020-05-05, 2020-06-06, 2020-07-17, 2020-08-15, 2020-08-17, 2020-09-30, 2020-10-01, 2020-10-02, 2020-10-03, 2020-10-09, 2020-12-25, 2021-01-01"""
name="""추석, 추석, 추석, 대체휴무일, 개천절, 한글날, 기독탄신일, 1월1일, 설날, 설날, 설날, 삼일절, 어린이날, 대체공휴일, 부처님오신날, 현충일, 제헌절, 광복절, 추석, 추석, 추석, 개천절, 한글날, 기독탄신일, 1월1일, 설날, 설날, 설날, 설날, 삼일절, 제21대 국회의원선거, 부처님오신날, 어린이날, 현충일, 제헌절, 광복절, 임시공휴일, 추석, 추석, 추석, 개천절, 한글날, 기독탄신일, 1월1일"""

date=date.split(',')
name=name.split(',')
holidays=pd.DataFrame({'date':date,'name':name})
holidays.head()
```

|      |       date |       name |
| ---: | ---------: | ---------: |
|    0 | 2018-09-23 |       추석 |
|    1 | 2018-09-24 |       추석 |
|    2 | 2018-09-25 |       추석 |
|    3 | 2018-09-26 | 대체휴무일 |
|    4 | 2018-10-03 |     개천절 |

```python
# 공휴일 train
train_holidays = train[train['date'].isin(holidays['date'])]

# 일반 train
train_not_holidays = train[~train['date'].isin(holidays['date'])]
```

#### 주말&휴일과 평일 분리

```python
from datetime import timedelta, date
def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
        
# train 시작하는 날
start_dt = date(2018,9,9)
# train 끝나는 날 -> new_train을 포함하면서 end_dt를 바꿈
end_dt = date(2020,12,8)

# sub 시작하는 날 -> new_train을 포함하면서 end_dt를 바꿈
start_sub = date(2020,12,9)
# sub 끝나는 날
end_sub = date(2021,1,8)

# 토요일과 일요일
weekdays = [5,6]

# train 정보 저장 : test 데이터로 검정할 때 확인
t_weekday = []
t_weekend = []
for dt in daterange(start_dt, end_dt):
    if dt.weekday() not in weekdays:               
        t_weekday.append(dt.strftime("%Y-%m-%d"))
    else:
        t_weekend.append(dt.strftime("%Y-%m-%d"))


# 제출할 때 적용하기 위함
sub_weekday = []
sub_weekend = []
for dt in daterange(start_sub, end_sub):
    if dt.weekday() not in weekdays:               
        sub_weekday.append(dt.strftime("%Y-%m-%d"))
    else:
        sub_weekend.append(dt.strftime("%Y-%m-%d"))
# train의 주말 연,월,일만 뽑기
train_5day = pd.to_datetime(np.array(t_weekday)).date
train_2day = pd.to_datetime(np.array(t_weekend)).date

# train의 주말 연,월,일만 뽑기
sub_5day = pd.to_datetime(np.array(sub_weekday)).date
sub_2day = pd.to_datetime(np.array(sub_weekend)).date

# submission의 date도 같은 형식으로 뽑기
submission['DateTime'] = pd.to_datetime(submission['DateTime']).dt.date
## train의 주말 연,월,일만 뽑기
train_5day = pd.to_datetime(np.array(t_weekday)).date
train_2day = pd.to_datetime(np.array(t_weekend)).date
holiday_date = pd.to_datetime(np.array(holidays['date'])).date


### 주말 및 공휴일과 평일을 분리하는 함수
    # train
train_holidays['date'] = pd.to_datetime(train_holidays['date'])

# train과 sub의 휴일 분리
t_holidays = holiday_date[holiday_date <= date(2020, 12, 8)]
sub_holidays = holiday_date[holiday_date > date(2020, 12, 8)]
all_train_rest = np.unique(np.sort(np.append(np.array(train_holidays['date'].dt.date), train_2day)))
all_sub_rest = np.sort(np.append(sub_holidays, sub_2day))

def data_rest_distribution(data, date, train_or_sub_date):
    data[date] = pd.to_datetime(data[date]).dt.date
    
    # 분할
    data_rest = data[data[date].isin(train_or_sub_date)]# [['date',col]]
    data_not_rest = data[~data[date].isin(train_or_sub_date)]# [['date',col]]
    
    return data_rest, data_not_rest
```

### 대회 개수의 영향력 확인

- 일별 진행중인 대회 개수가 다르고 그에 따른 참가자 수 또한 다르다.
- 진행중인 대회 수에 따른 Y의 분포 차이 확인
- 진행되는 대회 수가 많을 수록 Y가 모두 커짐

```python
## 필요한 변수만 남긴 뒤 결측치와 변수 제거
competition = competition_info[['period_start','period_end']].dropna()

# competition

# 각 날짜별 진행되고 있는 대회 개수
# competition 데이터셋의 모든 기간을 포함.

def date_cpt_count(competition):
    from datetime import date, timedelta

    # datatype: datetime으로 변경
    competition['period_start'] =  pd.to_datetime(competition.period_start)
    competition['period_end'] =  pd.to_datetime(competition.period_end)
    # date로 변경
    competition['period_start'] = competition.period_start.dt.date
    competition['period_end'] = competition.period_end.dt.date

    # 모든 대회 기간을 포함하는 datelist 생성. 897개
    start = competition['period_start'].min()
    end = competition['period_end'].max()

    datelist = []
    for i in range((end-start).days +1):
        datelist.append(start+timedelta(days=i))

    # 각 날짜에 대회가 몇개 열렸는지 매칭.
    cpt_cnt = [0 for i in range(len(datelist))] 

    for i in range(len(competition)):
        s = competition.iloc[i]['period_start']
        e = competition.iloc[i]['period_end']
      
        t = 0
        for d in datelist:
            if d >= s and d<=e:
                cpt_cnt[t]+=1
            t+=1

    date_cpt_cnt = pd.DataFrame({'date':datelist
                                ,'count': cpt_cnt})
    date_cpt_cnt.head()

    return date_cpt_cnt

competition = date_cpt_count(competition)

## 대회 개수 변수  추가
train['competition_counts'] = pd.Series(competition[(competition['date']>=date(2018,9,9)) <=date(2020, 12, 8))].reset_index(drop = True)['count'])
submission['competition_counts'] = pd.Series(competition[(competition['date']>=date(2020,12,9)) <=date(2021, 1, 8))].reset_index(drop = True)['count'])
```

#### 1) train 전체 기간으로 본 대회 개수 분포

```python
plt.figure(figsize = (20,5))
## 휴일과 평일을 나누지 않은 상태(train)의 대회 수에 따른 분포 확인
plt.subplot(1, 2, 1)
train.competition_counts.value_counts().plot(kind = 'bar')
plt.title('train')
plt.ylabel("Competition count")

## submission 기간 중의 대회 개수 확인
plt.subplot(1, 2, 2)
submission.competition_counts.value_counts().plot(kind = 'bar')
plt.title('submission')
plt.ylabel("Competition count")
Text(0, 0.5, 'Competition count')
```

![img](data:image/png;base64,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)

```python
plt.figure(figsize = (20,15))
plt.subplot(2, 2, 1)
sns.boxplot(x = 'competition_counts',
           y = '사용자',
           data = train)
plt.title('User')
plt.ylabel("User count")

plt.subplot(2, 2, 2)
sns.boxplot(x = 'competition_counts',
           y = '세션',
           data = train)
plt.title('Session')
plt.ylabel("Session count")

plt.subplot(2, 2, 3)
sns.boxplot(x = 'competition_counts',
           y = '신규방문자',
           data = train)
plt.title('New User')
plt.ylabel("New User count")

plt.subplot(2, 2, 4)
sns.boxplot(x = 'competition_counts',
           y = '페이지뷰',
           data = train)
plt.title('Page view')
plt.ylabel("Page view count")
Text(0, 0.5, 'Page view count')
```

![img](data:image/png;base64,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)

#### 2) 휴일의 대회 개수에 따른 분포

```python
## 데이터 분할
train_rest, train_not_rest = data_rest_distribution(train, 'date', all_train_rest)
sub_rest, sub_not_rest = data_rest_distribution(submission, 'DateTime', all_sub_rest)

train_rest = train_rest.set_index('date')
train_not_rest = train_not_rest.set_index('date')
sub_rest = sub_rest.set_index('DateTime')
sub_not_rest = sub_not_rest.set_index('DateTime')

print(train_rest.shape)
print(train_not_rest.shape)
print(sub_rest.shape)
print(sub_not_rest.shape)
(235, 8)
(587, 8)
(10, 5)
(21, 5)
plt.figure(figsize = (20,13))
plt.subplot(2, 2, 1)
sns.boxplot(x = 'competition_counts',
           y = '사용자',
           data = train_rest)
plt.title('User')
plt.ylabel("User count")

plt.subplot(2, 2, 2)
sns.boxplot(x = 'competition_counts',
           y = '세션',
           data = train_rest)
plt.title('Session')
plt.ylabel("Session count")

plt.subplot(2, 2, 3)
sns.boxplot(x = 'competition_counts',
           y = '신규방문자',
           data = train_rest)
plt.title('New User')
plt.ylabel("New User count")

plt.subplot(2, 2, 4)
sns.boxplot(x = 'competition_counts',
           y = '페이지뷰',
           data = train_rest)
plt.title('Page view')
plt.ylabel("Page view count")
Text(0, 0.5, 'Page view count')
```

![img](data:image/png;base64,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)

#### 3) 평일 대회개수에 따른 분포

```python
plt.figure(figsize = (20,13))
plt.subplot(2, 2, 1)
sns.boxplot(x = 'competition_counts',
           y = '사용자',
           data = train_not_rest)
plt.title('User')
plt.ylabel("User count")


plt.subplot(2, 2, 2)
sns.boxplot(x = 'competition_counts',
           y = '세션',
           data = train_not_rest)
plt.title('Session')
plt.ylabel("Session count")

plt.subplot(2, 2, 3)
sns.boxplot(x = 'competition_counts',
           y = '신규방문자',
           data = train_not_rest)
plt.title('New User')
plt.ylabel("New User count")

plt.subplot(2, 2, 4)
sns.boxplot(x = 'competition_counts',
           y = '페이지뷰',
           data = train_not_rest)
plt.title('Page view')
plt.ylabel("Page view count")
Text(0, 0.5, 'Page view count')
```

![img](data:image/png;base64,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)

# 4. LSTM

```python
train_lstm = train_data.copy()
# 날짜 관련 변수 생성 및 추가

def create_features(df):
    """
    Creates time series features from datetime index
    """

    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek #요일
    df['quarter'] = pd.to_datetime(df['date']).dt.quarter 
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['year'] = pd.to_datetime(df['date']).dt.year # 날짜의 해당 연도 시작에서부터의 일 수를 반환 
    df['dayofyear'] = pd.to_datetime(df['date']).dt.dayofyear
    df['dayofmonth'] = pd.to_datetime(df['date']).dt.day
    df['weekofyear'] = pd.to_datetime(df['date']).dt.weekofyear
    
    X = df[['사용자','세션','신규방문자','페이지뷰','date','dayofweek','quarter','month','year'
           ,'dayofyear','dayofmonth','weekofyear']]
    X.set_index('date', inplace = True)
    X.index = pd.to_datetime(X.index)
    
    return X

train_lstm = create_features(train_lstm)
train_lstm.head()
```

|            | 사용자 | 세션 | 신규방문자 | 페이지뷰 | dayofweek | quarter | month | year | dayofyear | dayofmonth | weekofyear |
| ---------: | -----: | ---: | ---------: | -------: | --------: | ------: | ----: | ---: | --------: | ---------: | ---------: |
|       date |        |      |            |          |           |         |       |      |           |            |            |
| 2018-09-09 |    281 |  266 |         73 |     1826 |         6 |       3 |     9 | 2018 |       252 |          9 |         36 |
| 2018-09-10 |    264 |  247 |         51 |     2092 |         0 |       3 |     9 | 2018 |       253 |         10 |         37 |
| 2018-09-11 |    329 |  310 |         58 |     1998 |         1 |       3 |     9 | 2018 |       254 |         11 |         37 |
| 2018-09-12 |    300 |  287 |         45 |     2595 |         2 |       3 |     9 | 2018 |       255 |         12 |         37 |
| 2018-09-13 |    378 |  344 |         50 |     3845 |         3 |       3 |     9 | 2018 |       256 |         13 |         37 |

```python
holidays['holiday']=1
holidays.set_index('date', inplace = True)
holidays.index = pd.to_datetime(holidays.index)

train_lstm.insert(4, 'holiday', holidays['holiday'])
train_lstm = train_lstm.fillna(0)

# date가 공휴일인 경우에 페이지뷰의 수를 10% 감소시키기 

for i, value in enumerate(train_lstm['holiday']):
    if value==1:
    train_lstm['페이지뷰'][i]= train_lstm['페이지뷰'][i] - (train_lstm['페이지뷰'][i]*0.1)  

train_lstm = train_lstm.drop(['holiday'],axis=1)
```

- 범주형 인코딩

```python
# bianry encoding
encoder = ce.BinaryEncoder(cols=['dayofyear'])
train_lstm = encoder.fit_transform(train_lstm)

encoder = ce.BinaryEncoder(cols=['dayofmonth'])
train_lstm = encoder.fit_transform(train_lstm)

encoder = ce.BinaryEncoder(cols=['weekofyear'])
train_lstm = encoder.fit_transform(train_lstm)
# one-hot encoding

def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix = column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["dayofweek", "quarter", "month", "year"]
train_lstm = dummy_data(train_lstm,dummy_columns)

print(train_lstm.shape)
train_lstm.head(3)
(822, 53)
```

|            | 사용자 | 세션 | 신규방문자 | 페이지뷰 | dayofyear_0 | dayofyear_1 | dayofyear_2 | dayofyear_3 | dayofyear_4 | dayofyear_5 | dayofyear_6 | dayofyear_7 | dayofyear_8 | dayofyear_9 | dayofmonth_0 | dayofmonth_1 | dayofmonth_2 | dayofmonth_3 | dayofmonth_4 | dayofmonth_5 | weekofyear_0 | weekofyear_1 | weekofyear_2 | weekofyear_3 | weekofyear_4 | weekofyear_5 | weekofyear_6 | dayofweek_0 | dayofweek_1 | dayofweek_2 | dayofweek_3 | dayofweek_4 | dayofweek_5 | dayofweek_6 | quarter_1 | quarter_2 | quarter_3 | quarter_4 | month_1 | month_2 | month_3 | month_4 | month_5 | month_6 | month_7 | month_8 | month_9 | month_10 | month_11 | month_12 | year_2018 | year_2019 | year_2020 |
| ---------: | -----: | ---: | ---------: | -------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | --------: | --------: | --------: | --------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | -------: | -------: | -------: | --------: | --------: | --------: |
|       date |        |      |            |          |             |             |             |             |             |             |             |             |             |             |              |              |              |              |              |              |              |              |              |              |              |              |              |             |             |             |             |             |             |             |           |           |           |           |         |         |         |         |         |         |         |         |         |          |          |          |           |           |           |
| 2018-09-09 |    281 |  266 |         73 |     1826 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |            1 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |
| 2018-09-10 |    264 |  247 |         51 |     2092 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           0 |            0 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |
| 2018-09-11 |    329 |  310 |         58 |     1998 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           1 |            0 |            0 |            0 |            0 |            1 |            1 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |           0 |           1 |           0 |           0 |           0 |           0 |           0 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |

```python
### scaling
mini = train_lstm.iloc[:,:4].min()
size = train_lstm.iloc[:,:4].max() - train_lstm.iloc[:,:4].min()
train_lstm.iloc[:,:4] = (train_lstm.iloc[:,:4] -  mini) / size
train_lstm.head()
```

|            |   사용자 |     세션 | 신규방문자 | 페이지뷰 | dayofyear_0 | dayofyear_1 | dayofyear_2 | dayofyear_3 | dayofyear_4 | dayofyear_5 | dayofyear_6 | dayofyear_7 | dayofyear_8 | dayofyear_9 | dayofmonth_0 | dayofmonth_1 | dayofmonth_2 | dayofmonth_3 | dayofmonth_4 | dayofmonth_5 | weekofyear_0 | weekofyear_1 | weekofyear_2 | weekofyear_3 | weekofyear_4 | weekofyear_5 | weekofyear_6 | dayofweek_0 | dayofweek_1 | dayofweek_2 | dayofweek_3 | dayofweek_4 | dayofweek_5 | dayofweek_6 | quarter_1 | quarter_2 | quarter_3 | quarter_4 | month_1 | month_2 | month_3 | month_4 | month_5 | month_6 | month_7 | month_8 | month_9 | month_10 | month_11 | month_12 | year_2018 | year_2019 | year_2020 |
| ---------: | -------: | -------: | ---------: | -------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | --------: | --------: | --------: | --------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | -------: | -------: | -------: | --------: | --------: | --------: |
|       date |          |          |            |          |             |             |             |             |             |             |             |             |             |             |              |              |              |              |              |              |              |              |              |              |              |              |              |             |             |             |             |             |             |             |           |           |           |           |         |         |         |         |         |         |         |         |         |          |          |          |           |           |           |
| 2018-09-09 | 0.051689 | 0.049083 |   0.042604 | 0.011735 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |            1 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |
| 2018-09-10 | 0.048551 | 0.045564 |   0.029586 | 0.013446 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           0 |            0 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |           1 |           0 |           0 |           0 |           0 |           0 |           0 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |
| 2018-09-11 | 0.060550 | 0.057233 |   0.033728 | 0.012842 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           1 |            0 |            0 |            0 |            0 |            1 |            1 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |           0 |           1 |           0 |           0 |           0 |           0 |           0 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |
| 2018-09-12 | 0.055197 | 0.052973 |   0.026036 | 0.016682 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           0 |           0 |            0 |            0 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |           0 |           0 |           1 |           0 |           0 |           0 |           0 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |
| 2018-09-13 | 0.069596 | 0.063530 |   0.028994 | 0.024724 |           0 |           0 |           0 |           0 |           0 |           0 |           0 |           1 |           0 |           1 |            0 |            0 |            0 |            1 |            0 |            1 |            0 |            0 |            0 |            0 |            0 |            1 |            0 |           0 |           0 |           0 |           1 |           0 |           0 |           0 |         0 |         0 |         1 |         0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       0 |       1 |        0 |        0 |        0 |         1 |         0 |         0 |

```python
input_window = 61
output_window = 61 # 7

window_x = np.zeros((train_lstm.shape[0] - (input_window + output_window), input_window, 53))
window_y = np.zeros((train_lstm.shape[0] - (input_window + output_window), output_window, 4))

for start in range(train_lstm.shape[0] - (input_window + output_window)):
    end = start + input_window    
    window_x[start,:, :] = train_lstm.iloc[start : end                , : ].values
    window_y[start,:, :] = train_lstm.iloc[end   : end + output_window, :4 ].values


print('window_x.shape: ', window_x.shape)
print('window_y.shape: ', window_y.shape)
window_x.shape:  (700, 61, 53)
window_y.shape:  (700, 61, 4)
tf.random.set_seed(40)


model = Sequential()
model.add(LSTM(32, input_shape=(61, 53), return_sequences=True))  # (timestep, features)
model.add(LSTM(32, return_sequences=True)) 
model.add(LSTM(32, return_sequences=True)) 
model.add(TimeDistributed(Dense(4)))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=opt)

model.fit(window_x, window_y, epochs=100, batch_size=50)
Epoch 1/100
14/14 [==============================] - 4s 56ms/step - loss: 0.0436
Epoch 2/100
14/14 [==============================] - 1s 58ms/step - loss: 0.0112
Epoch 3/100
14/14 [==============================] - 1s 57ms/step - loss: 0.0085
Epoch 4/100
14/14 [==============================] - 1s 56ms/step - loss: 0.0073
Epoch 5/100
14/14 [==============================] - 1s 57ms/step - loss: 0.0064
Epoch 6/100
14/14 [==============================] - 1s 56ms/step - loss: 0.0055
Epoch 7/100
14/14 [==============================] - 1s 56ms/step - loss: 0.0045
Epoch 8/100
14/14 [==============================] - 1s 61ms/step - loss: 0.0039
Epoch 9/100
14/14 [==============================] - 1s 58ms/step - loss: 0.0035
Epoch 10/100
14/14 [==============================] - 1s 58ms/step - loss: 0.0034
Epoch 11/100
14/14 [==============================] - 1s 56ms/step - loss: 0.0029
Epoch 12/100
14/14 [==============================] - 1s 56ms/step - loss: 0.0026
Epoch 13/100
14/14 [==============================] - 1s 57ms/step - loss: 0.0023
Epoch 14/100
14/14 [==============================] - 1s 58ms/step - loss: 0.0021
Epoch 15/100
14/14 [==============================] - 1s 57ms/step - loss: 0.0019
Epoch 16/100
14/14 [==============================] - 1s 56ms/step - loss: 0.0018
Epoch 17/100
14/14 [==============================] - 1s 59ms/step - loss: 0.0016
Epoch 18/100
14/14 [==============================] - 1s 57ms/step - loss: 0.0014
Epoch 19/100
14/14 [==============================] - 1s 58ms/step - loss: 0.0013
Epoch 20/100
14/14 [==============================] - 1s 58ms/step - loss: 0.0012
Epoch 21/100
14/14 [==============================] - 1s 59ms/step - loss: 0.0012
Epoch 22/100
14/14 [==============================] - 1s 61ms/step - loss: 0.0011
Epoch 23/100
14/14 [==============================] - 1s 60ms/step - loss: 0.0010
Epoch 24/100
14/14 [==============================] - 1s 60ms/step - loss: 8.7813e-04
Epoch 25/100
14/14 [==============================] - 1s 58ms/step - loss: 7.9791e-04
Epoch 26/100
14/14 [==============================] - 1s 59ms/step - loss: 7.3549e-04
Epoch 27/100
14/14 [==============================] - 1s 61ms/step - loss: 6.8834e-04
Epoch 28/100
14/14 [==============================] - 1s 60ms/step - loss: 6.8159e-04
Epoch 29/100
14/14 [==============================] - 1s 61ms/step - loss: 6.0269e-04
Epoch 30/100
14/14 [==============================] - 1s 60ms/step - loss: 5.6324e-04
Epoch 31/100
14/14 [==============================] - 1s 59ms/step - loss: 5.6658e-04
Epoch 32/100
14/14 [==============================] - 1s 60ms/step - loss: 5.2296e-04
Epoch 33/100
14/14 [==============================] - 1s 59ms/step - loss: 4.8611e-04
Epoch 34/100
14/14 [==============================] - 1s 60ms/step - loss: 5.3483e-04
Epoch 35/100
14/14 [==============================] - 1s 60ms/step - loss: 5.5414e-04
Epoch 36/100
14/14 [==============================] - 1s 60ms/step - loss: 4.3234e-04
Epoch 37/100
14/14 [==============================] - 1s 59ms/step - loss: 3.8763e-04
Epoch 38/100
14/14 [==============================] - 1s 58ms/step - loss: 3.6034e-04
Epoch 39/100
14/14 [==============================] - 1s 59ms/step - loss: 3.3455e-04
Epoch 40/100
14/14 [==============================] - 1s 59ms/step - loss: 3.3514e-04
Epoch 41/100
14/14 [==============================] - 1s 59ms/step - loss: 3.3761e-04
Epoch 42/100
14/14 [==============================] - 1s 61ms/step - loss: 3.3991e-04
Epoch 43/100
14/14 [==============================] - 1s 59ms/step - loss: 3.0050e-04
Epoch 44/100
14/14 [==============================] - 1s 59ms/step - loss: 3.0875e-04
Epoch 45/100
14/14 [==============================] - 1s 60ms/step - loss: 3.0443e-04
Epoch 46/100
14/14 [==============================] - 1s 59ms/step - loss: 2.7491e-04
Epoch 47/100
14/14 [==============================] - 1s 59ms/step - loss: 2.5329e-04
Epoch 48/100
14/14 [==============================] - 1s 59ms/step - loss: 2.6252e-04
Epoch 49/100
14/14 [==============================] - 1s 59ms/step - loss: 2.7986e-04
Epoch 50/100
14/14 [==============================] - 1s 60ms/step - loss: 4.1914e-04
Epoch 51/100
14/14 [==============================] - 1s 59ms/step - loss: 2.7060e-04
Epoch 52/100
14/14 [==============================] - 1s 60ms/step - loss: 2.5094e-04
Epoch 53/100
14/14 [==============================] - 1s 61ms/step - loss: 2.2887e-04
Epoch 54/100
14/14 [==============================] - 1s 60ms/step - loss: 2.2013e-04
Epoch 55/100
14/14 [==============================] - 1s 60ms/step - loss: 2.1978e-04
Epoch 56/100
14/14 [==============================] - 1s 59ms/step - loss: 2.3421e-04
Epoch 57/100
14/14 [==============================] - 1s 61ms/step - loss: 2.1953e-04
Epoch 58/100
14/14 [==============================] - 1s 60ms/step - loss: 2.1194e-04
Epoch 59/100
14/14 [==============================] - 1s 61ms/step - loss: 2.2164e-04
Epoch 60/100
14/14 [==============================] - 1s 60ms/step - loss: 2.1350e-04
Epoch 61/100
14/14 [==============================] - 1s 59ms/step - loss: 2.0322e-04
Epoch 62/100
14/14 [==============================] - 1s 61ms/step - loss: 1.8663e-04
Epoch 63/100
14/14 [==============================] - 1s 60ms/step - loss: 2.2056e-04
Epoch 64/100
14/14 [==============================] - 1s 60ms/step - loss: 2.4533e-04
Epoch 65/100
14/14 [==============================] - 1s 61ms/step - loss: 3.6015e-04
Epoch 66/100
14/14 [==============================] - 1s 61ms/step - loss: 2.7461e-04
Epoch 67/100
14/14 [==============================] - 1s 61ms/step - loss: 2.9495e-04
Epoch 68/100
14/14 [==============================] - 1s 59ms/step - loss: 2.3474e-04
Epoch 69/100
14/14 [==============================] - 1s 60ms/step - loss: 1.7854e-04
Epoch 70/100
14/14 [==============================] - 1s 62ms/step - loss: 1.5511e-04
Epoch 71/100
14/14 [==============================] - 1s 61ms/step - loss: 1.5153e-04
Epoch 72/100
14/14 [==============================] - 1s 59ms/step - loss: 1.4544e-04
Epoch 73/100
14/14 [==============================] - 1s 60ms/step - loss: 1.5204e-04
Epoch 74/100
14/14 [==============================] - 1s 60ms/step - loss: 1.3232e-04
Epoch 75/100
14/14 [==============================] - 1s 60ms/step - loss: 1.2821e-04
Epoch 76/100
14/14 [==============================] - 1s 60ms/step - loss: 1.3898e-04
Epoch 77/100
14/14 [==============================] - 1s 59ms/step - loss: 1.2826e-04
Epoch 78/100
14/14 [==============================] - 1s 60ms/step - loss: 1.2939e-04
Epoch 79/100
14/14 [==============================] - 1s 62ms/step - loss: 1.2026e-04
Epoch 80/100
14/14 [==============================] - 1s 60ms/step - loss: 1.2118e-04
Epoch 81/100
14/14 [==============================] - 1s 61ms/step - loss: 1.2975e-04
Epoch 82/100
14/14 [==============================] - 1s 59ms/step - loss: 1.1680e-04
Epoch 83/100
14/14 [==============================] - 1s 60ms/step - loss: 1.3297e-04
Epoch 84/100
14/14 [==============================] - 1s 59ms/step - loss: 1.2292e-04
Epoch 85/100
14/14 [==============================] - 1s 59ms/step - loss: 1.2571e-04
Epoch 86/100
14/14 [==============================] - 1s 58ms/step - loss: 1.1242e-04
Epoch 87/100
14/14 [==============================] - 1s 61ms/step - loss: 1.1258e-04
Epoch 88/100
14/14 [==============================] - 1s 60ms/step - loss: 1.1171e-04
Epoch 89/100
14/14 [==============================] - 1s 60ms/step - loss: 1.0343e-04
Epoch 90/100
14/14 [==============================] - 1s 59ms/step - loss: 9.8354e-05
Epoch 91/100
14/14 [==============================] - 1s 58ms/step - loss: 1.0535e-04
Epoch 92/100
14/14 [==============================] - 1s 62ms/step - loss: 9.7086e-05
Epoch 93/100
14/14 [==============================] - 1s 59ms/step - loss: 1.0235e-04
Epoch 94/100
14/14 [==============================] - 1s 59ms/step - loss: 1.0555e-04
Epoch 95/100
14/14 [==============================] - 1s 61ms/step - loss: 1.1828e-04
Epoch 96/100
14/14 [==============================] - 1s 59ms/step - loss: 1.1794e-04
Epoch 97/100
14/14 [==============================] - 1s 59ms/step - loss: 1.0543e-04
Epoch 98/100
14/14 [==============================] - 1s 59ms/step - loss: 9.8751e-05
Epoch 99/100
14/14 [==============================] - 1s 59ms/step - loss: 1.0161e-04
Epoch 100/100
14/14 [==============================] - 1s 59ms/step - loss: 1.2607e-04
<tensorflow.python.keras.callbacks.History at 0x7f02cb875b70>
last_df = train_lstm.iloc[-61:,:].values[np.newaxis,...] 
lstm_pred = model.predict(last_df)

lstm_pred = lstm_pred.reshape(output_window,4)
lstm_pred = lstm_pred * size[:4].values + mini[:4].values
lstm_pred = lstm_pred.astype(int)
lstm_pred = lstm_pred[:31]
lstm_pred
array([[  2186,   2147,    490,  53605],
       [  3065,   3035,    761,  71873],
       [  3682,   3573,    957, 101012],
       [  2105,   2018,    484,  49675],
       [  2007,   1956,    429,  47148],
       [  3382,   3325,    816,  89366],
       [  2937,   2868,    694,  77382],
       [  3437,   3379,    817,  91721],
       [  3312,   3262,    781,  80183],
       [  3155,   3145,    787,  75434],
       [  2046,   2065,    552,  45086],
       [  2108,   2155,    537,  50554],
       [  3697,   3743,   1040, 104187],
       [  2632,   2612,    602,  66276],
       [  2566,   2600,    636,  61461],
       [  2961,   3017,    829,  68332],
       [  2973,   3013,    837,  76922],
       [  1563,   1586,    437,  33327],
       [  1458,   1527,    405,  32977],
       [  2815,   2907,    773,  73740],
       [  2485,   2558,    688,  67564],
       [  2073,   2181,    580,  55033],
       [  2529,   2703,    724,  68668],
       [  2051,   2215,    632,  56342],
       [  1955,   2093,    575,  55401],
       [  2107,   2210,    499,  61820],
       [  3036,   3150,    669,  89636],
       [  2717,   2768,    539,  77047],
       [  3319,   3391,    675,  92410],
       [  3367,   3416,    753,  84647],
       [  2784,   2806,    655,  61985]])
lstm_result = pd.DataFrame(lstm_pred) 
lstm_result=lstm_result.rename({0:"사용자",1:"세션",2:"신규방문자",3:"페이지뷰"},axis="columns") 
lstm_result.set_index(submission['DateTime'], inplace=True)
lstm_result
```

|            | 사용자 | 세션 | 신규방문자 | 페이지뷰 |
| ---------: | -----: | ---: | ---------: | -------: |
|   DateTime |        |      |            |          |
| 2020-12-09 |   2186 | 2147 |        490 |    53605 |
| 2020-12-10 |   3065 | 3035 |        761 |    71873 |
| 2020-12-11 |   3682 | 3573 |        957 |   101012 |
| 2020-12-12 |   2105 | 2018 |        484 |    49675 |
| 2020-12-13 |   2007 | 1956 |        429 |    47148 |
| 2020-12-14 |   3382 | 3325 |        816 |    89366 |
| 2020-12-15 |   2937 | 2868 |        694 |    77382 |
| 2020-12-16 |   3437 | 3379 |        817 |    91721 |
| 2020-12-17 |   3312 | 3262 |        781 |    80183 |
| 2020-12-18 |   3155 | 3145 |        787 |    75434 |
| 2020-12-19 |   2046 | 2065 |        552 |    45086 |
| 2020-12-20 |   2108 | 2155 |        537 |    50554 |
| 2020-12-21 |   3697 | 3743 |       1040 |   104187 |
| 2020-12-22 |   2632 | 2612 |        602 |    66276 |
| 2020-12-23 |   2566 | 2600 |        636 |    61461 |
| 2020-12-24 |   2961 | 3017 |        829 |    68332 |
| 2020-12-25 |   2973 | 3013 |        837 |    76922 |
| 2020-12-26 |   1563 | 1586 |        437 |    33327 |
| 2020-12-27 |   1458 | 1527 |        405 |    32977 |
| 2020-12-28 |   2815 | 2907 |        773 |    73740 |
| 2020-12-29 |   2485 | 2558 |        688 |    67564 |
| 2020-12-30 |   2073 | 2181 |        580 |    55033 |
| 2020-12-31 |   2529 | 2703 |        724 |    68668 |
| 2021-01-01 |   2051 | 2215 |        632 |    56342 |
| 2021-01-02 |   1955 | 2093 |        575 |    55401 |
| 2021-01-03 |   2107 | 2210 |        499 |    61820 |
| 2021-01-04 |   3036 | 3150 |        669 |    89636 |
| 2021-01-05 |   2717 | 2768 |        539 |    77047 |
| 2021-01-06 |   3319 | 3391 |        675 |    92410 |
| 2021-01-07 |   3367 | 3416 |        753 |    84647 |
| 2021-01-08 |   2784 | 2806 |        655 |    61985 |

# 5. XGBOOST

- 평일과 휴일 분리 + 대회 개수에 따른 분리 XGB 적용
- 휴일의 경우 : [0,1,2], [3,4,5,6,7]
- 주중의 경우 : [0,1,2], [3,4,5,6,7]

```python
## EDA를 통해 파악하여 휴일과 같은 패턴을 보이는 날들 공휴일에 추가
holidays.reset_index(inplace = True)
holidays['date'] = pd.to_datetime(holidays['date']).dt.date
holidays = holidays.append({'date' : date(2019,5,2), 'name':'연휴'}, ignore_index = True)
holidays = holidays.append({'date' : date(2019,5,3), 'name':'연휴'}, ignore_index = True)
holidays = holidays.append({'date' : date(2020,5,7), 'name':'연휴'}, ignore_index = True)
holidays = holidays.append({'date' : date(2020,5,4), 'name':'연휴'}, ignore_index = True)
holidays = holidays.append({'date' : date(2020,5,1), 'name':'연휴'}, ignore_index = True)
holidays = holidays.append({'date' : date(2019,12,31), 'name':'연말'}, ignore_index = True)
holidays = holidays.append({'date' : date(2020,12,31), 'name':'연말'}, ignore_index = True)
## train의 주말 연,월,일만 뽑기
train_5day = pd.to_datetime(np.array(t_weekday)).date
train_2day = pd.to_datetime(np.array(t_weekend)).date
holiday_date = pd.to_datetime(np.array(holidays['date'])).date

# train의 주말 연,월,일만 뽑기
train_5day = pd.to_datetime(np.array(t_weekday)).date
train_2day = pd.to_datetime(np.array(t_weekend)).date

# train의 주말 연,월,일만 뽑기
sub_5day = pd.to_datetime(np.array(sub_weekday)).date
sub_2day = pd.to_datetime(np.array(sub_weekend)).date

# submission의 date도 같은 형식으로 뽑기
submission['DateTime'] = pd.to_datetime(submission['DateTime']).dt.date

# 공휴일 train
train_holidays = train[train['date'].isin(holidays['date'])]
# 일반 train
train_not_holidays = train[~train['date'].isin(holidays['date'])]


################ 주말 및 공휴일과 평일을 분리하는 함수 ###################
    # train
train_holidays['date'] = pd.to_datetime(train_holidays['date'])

# train과 sub의 휴일 분리
t_holidays = holiday_date[holiday_date <= date(2020, 12, 8)]
sub_holidays = holiday_date[holiday_date > date(2020, 12, 8)]
all_train_rest = np.unique(np.sort(np.append(np.array(train_holidays['date'].dt.date), train_2day)))
all_sub_rest = np.sort(np.append(sub_holidays, sub_2day))

def data_rest_distribution(data, date, train_or_sub_date):
    data[date] = pd.to_datetime(data[date]).dt.date
    
    # 분할
    data_rest = data[data[date].isin(train_or_sub_date)]# [['date',col]]
    data_not_rest = data[~data[date].isin(train_or_sub_date)]# [['date',col]]
    
    return data_rest, data_not_rest
## 여러 정보를 바탕으로 XGB에는 19년 5월 부터 데이터 사용
## 여러 정보를 바탕으로 XGB에는 19년 5월 부터 데이터 사용
xgb_train = train[(pd.to_datetime(train['date'])>='2019-05-01') <='2020-12-08')]
## 데이터 분할
# 대회 정보 추가
xgb_train['competition_counts'] = pd.Series(competition[(competition['date']>=date(2018,9,9)) <=date(2020, 12, 8))].reset_index(drop = True)['count'])
submission['competition_counts'] = pd.Series(competition[(competition['date']>=date(2020,12,9)) <=date(2021, 1, 8))].reset_index(drop = True)['count'])

# 분할
train_rest, train_not_rest = data_rest_distribution(xgb_train, 'date', all_train_rest)
sub_rest, sub_not_rest = data_rest_distribution(submission, 'DateTime', all_sub_rest)

## 인덱스 제거
train_not_rest.reset_index(drop=True, inplace = True)
train_rest.reset_index(drop=True, inplace = True)
sub_rest.reset_index(drop=True, inplace = True)
sub_not_rest.reset_index(drop=True, inplace = True)

print(train_rest.shape)
print(train_not_rest.shape)
print(sub_rest.shape)
print(sub_not_rest.shape)
(195, 9)
(393, 9)
(11, 6)
(20, 6)
## 사용자 분리 : 휴일
t_user_rest_012 = train_rest[train_rest.competition_counts.isin([0,1,2])][['date','사용자']].reset_index(drop=True)
t_user_rest_34567 = train_rest[train_rest.competition_counts.isin([3,4,5,6,7])][['date','사용자']].reset_index(drop=True)
## 사용자 분리 : 평일
t_user_not_rest_012 = train_not_rest[train_not_rest.competition_counts.isin([0,1,2])][['date','사용자']].reset_index(drop=True)
t_user_not_rest_34567 = train_not_rest[train_not_rest.competition_counts.isin([3,4,5,6,7,9])][['date','사용자']].reset_index(drop=True)


## 세션 분리 : 휴일
t_sess_rest_012 = train_rest[train_rest.competition_counts.isin([0,1,2])][['date','세션']].reset_index(drop=True)
t_sess_rest_34567 = train_rest[train_rest.competition_counts.isin([3,4,5,6,7])][['date','세션']].reset_index(drop=True)
## 세션 분리 : 평일
t_sess_not_rest_012 = train_not_rest[train_not_rest.competition_counts.isin([0,1,2])][['date','세션']].reset_index(drop=True)
t_sess_not_rest_34567 = train_not_rest[train_not_rest.competition_counts.isin([3,4,5,6,7,9])][['date','세션']].reset_index(drop=True)


## 신규방문자 분리 : 휴일
t_new_rest_012 = train_rest[train_rest.competition_counts.isin([0,1,2])][['date','신규방문자']].reset_index(drop=True)
t_new_rest_34567 = train_rest[train_rest.competition_counts.isin([3,4,5,6,7])][['date','신규방문자']].reset_index(drop=True)
## 신규방문자 분리 : 평일
t_new_not_rest_012 = train_not_rest[train_not_rest.competition_counts.isin([0,1,2])][['date','신규방문자']].reset_index(drop=True)
t_new_not_rest_34567 = train_not_rest[train_not_rest.competition_counts.isin([3,4,5,6,7,9])][['date','신규방문자']].reset_index(drop=True)


## 페이지뷰 분리 : 휴일
t_page_rest_012 = train_rest[train_rest.competition_counts.isin([0,1,2])][['date','페이지뷰']].reset_index(drop=True)
t_page_rest_34567 = train_rest[train_rest.competition_counts.isin([3,4,5,6,7])][['date','페이지뷰']].reset_index(drop=True)
## 페이지뷰 분리 : 평일
t_page_not_rest_012 = train_not_rest[train_not_rest.competition_counts.isin([0,1,2])][['date','페이지뷰']].reset_index(drop=True)
t_page_not_rest_34567 = train_not_rest[train_not_rest.competition_counts.isin([3,4,5,6,7,9])][['date','페이지뷰']].reset_index(drop=True)
########################################################################################################################
#################################################### sub 분리 ##########################################################
########################################################################################################################
## 사용자 분리 : 휴일
s_user_rest_012 = sub_rest[sub_rest.competition_counts.isin([0,1,2])][['DateTime','사용자']].reset_index(drop=True)
s_user_rest_34567 = sub_rest[sub_rest.competition_counts.isin([3,4,5,6,7])][['DateTime','사용자']].reset_index(drop=True)
## 사용자 분리 : 평일
s_user_not_rest_012 = sub_not_rest[sub_not_rest.competition_counts.isin([0,1,2])][['DateTime','사용자']].reset_index(drop=True)
s_user_not_rest_34567 = sub_not_rest[sub_not_rest.competition_counts.isin([3,4,5,6,7,9])][['DateTime','사용자']].reset_index(drop=True)


## 세션 분리 : 휴일
s_sess_rest_012 = sub_rest[sub_rest.competition_counts.isin([0,1,2])][['DateTime','세션']].reset_index(drop=True)
s_sess_rest_34567= sub_rest[sub_rest.competition_counts.isin([3,4,5,6,7,9])][['DateTime','세션']].reset_index(drop=True)
## 세션 분리 : 평일
s_sess_not_rest_012 = sub_not_rest[sub_not_rest.competition_counts.isin([0,1,2])][['DateTime','세션']].reset_index(drop=True)
s_sess_not_rest_34567 = sub_not_rest[sub_not_rest.competition_counts.isin([3,4,5,6,7,9])][['DateTime','세션']].reset_index(drop=True)


## 신규방문자 분리 : 휴일
s_new_rest_012 = sub_rest[sub_rest.competition_counts.isin([0,1,2])][['DateTime','신규방문자']].reset_index(drop=True)
s_new_rest_34567 = sub_rest[sub_rest.competition_counts.isin([3,4,5,6,7])][['DateTime','신규방문자']].reset_index(drop=True)
## 신규방문자 분리 : 평일
s_new_not_rest_012 = sub_not_rest[sub_not_rest.competition_counts.isin([0,1,2])][['DateTime','신규방문자']].reset_index(drop=True)
s_new_not_rest_34567 = sub_not_rest[sub_not_rest.competition_counts.isin([3,4,5,6,7,9])][['DateTime','신규방문자']].reset_index(drop=True)


## 페이지뷰 분리 : 휴일
s_page_rest_012 = sub_rest[sub_rest.competition_counts.isin([0,1,2])][['DateTime','페이지뷰']].reset_index(drop=True)
s_page_rest_34567 = sub_rest[sub_rest.competition_counts.isin([3,4,5,6,7])][['DateTime','페이지뷰']].reset_index(drop=True)
## 페이지뷰 분리 : 평일
s_page_not_rest_012 = sub_not_rest[sub_not_rest.competition_counts.isin([0,1,2])][['DateTime','페이지뷰']].reset_index(drop=True)
s_page_not_rest_34567 = sub_not_rest[sub_not_rest.competition_counts.isin([3,4,5,6,7,9])][['DateTime','페이지뷰']].reset_index(drop=True)
```

- 알고리즘 적용

```python
print('휴일 사용자')
print(t_user_rest_012.shape, t_user_rest_34567.shape)
print('휴일 세션')
print(t_sess_rest_012.shape, t_sess_rest_34567.shape)
print('휴일 신규방문자')
print(t_new_rest_012.shape, t_new_rest_34567.shape)
print('휴일 페이지뷰')
print(t_page_rest_012.shape, t_page_rest_34567.shape, end = '\n\n')
print('주중 사용자')
print(t_user_not_rest_012.shape, t_user_not_rest_34567.shape)
print('주중 세션')
print(t_sess_not_rest_012.shape, t_sess_not_rest_34567.shape)
print('주중 신규방문자')
print(t_new_not_rest_012.shape, t_new_not_rest_34567.shape)
print('주중 페이지뷰')
print(t_page_not_rest_012.shape, t_page_not_rest_34567.shape)
휴일 사용자
(73, 2) (122, 2)
휴일 세션
(73, 2) (122, 2)
휴일 신규방문자
(73, 2) (122, 2)
휴일 페이지뷰
(73, 2) (122, 2)

주중 사용자
(144, 2) (249, 2)
주중 세션
(144, 2) (249, 2)
주중 신규방문자
(144, 2) (249, 2)
주중 페이지뷰
(144, 2) (249, 2)
print('휴일 사용자')
print(s_user_rest_012.shape, s_user_rest_34567.shape)
print('휴일 세션')
print(s_sess_rest_012.shape, s_sess_rest_34567.shape)
print('휴일 신규방문자')
print(s_new_rest_012.shape, s_new_rest_34567.shape)
print('휴일 페이지뷰')
print(s_page_rest_012.shape, s_page_rest_34567.shape, end = '\n\n')
print('주중 사용자')
print(s_user_not_rest_012.shape, s_user_not_rest_34567.shape)
print('주중 세션')
print(s_sess_not_rest_012.shape, s_sess_not_rest_34567.shape)
print('주중 신규방문자')
print(s_new_not_rest_012.shape, s_new_not_rest_34567.shape)
print('주중 페이지뷰')
print(s_page_not_rest_012.shape, s_page_not_rest_34567.shape)
휴일 사용자
(2, 2) (9, 2)
휴일 세션
(2, 2) (9, 2)
휴일 신규방문자
(2, 2) (9, 2)
휴일 페이지뷰
(2, 2) (9, 2)

주중 사용자
(2, 2) (18, 2)
주중 세션
(2, 2) (18, 2)
주중 신규방문자
(2, 2) (18, 2)
주중 페이지뷰
(2, 2) (18, 2)
## time_series로 만들기
def date_to_values(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # 입력 sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # 출력 sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

## 모델 fitting
def xgboost_forcasting(dataname, n_in_size):
    values = list(dataname.iloc[:,1].values)
    data = date_to_values(values, n_in = n_in_size)
    train_x, train_y = data[:,:-1], data[:,-1]
    xgboost = XGBRegressor(objective='reg:squarederror', 
                      n_estimators=1000, 
                      random_state = 40)
    xgboost.fit(train_x, train_y)
    return train_x, xgboost

## 예측 함수
def xgb_prediction(data, data2, model_name, num1, num2, name):
    for i in range(0,num2):
        if i==0:
            first = np.reshape(data[-1], (1,num1))
            pred = model_name.predict(first)
            data2[name][i] = pred
            tt = np.reshape(np.append(data[-1,1:], pred), (1,num1))
        else:
            pred = model_name.predict(tt)
            data2[name][i] = pred
            tt = np.reshape(np.append(tt[-1,1:], pred), (1,num1))
## 휴일 예측
train_x, xgboost = xgboost_forcasting(t_user_rest_012, 60)
xgb_prediction(train_x, s_user_rest_012 ,xgboost, 60, 2, '사용자')
train_x, xgboost = xgboost_forcasting(t_sess_rest_012, 60)
xgb_prediction(train_x, s_sess_rest_012 ,xgboost, 60, 2, '세션')
train_x, xgboost = xgboost_forcasting(t_new_rest_012, 60)
xgb_prediction(train_x, s_new_rest_012 ,xgboost, 60, 2, '신규방문자')
train_x, xgboost = xgboost_forcasting(t_page_rest_012, 60)
xgb_prediction(train_x, s_page_rest_012 ,xgboost, 60, 2, '페이지뷰')


train_x, xgboost = xgboost_forcasting(t_user_rest_34567, 90)
xgb_prediction(train_x, s_user_rest_34567 ,xgboost, 90, 9, '사용자')
train_x, xgboost = xgboost_forcasting(t_sess_rest_34567, 90)
xgb_prediction(train_x, s_sess_rest_34567 ,xgboost, 90, 9, '세션')
train_x, xgboost = xgboost_forcasting(t_new_rest_34567, 90)
xgb_prediction(train_x, s_new_rest_34567 ,xgboost, 90, 9, '신규방문자')
train_x, xgboost = xgboost_forcasting(t_page_rest_34567, 90)
xgb_prediction(train_x, s_page_rest_34567 ,xgboost, 90, 9, '페이지뷰')
## 평일 예측
train_x, xgboost = xgboost_forcasting(t_user_not_rest_012, 100)
xgb_prediction(train_x, s_user_not_rest_012 ,xgboost, 100, 2, '사용자')
train_x, xgboost = xgboost_forcasting(t_sess_not_rest_012, 100)
xgb_prediction(train_x, s_sess_not_rest_012 ,xgboost, 100, 2, '세션')
train_x, xgboost = xgboost_forcasting(t_new_not_rest_012, 100)
xgb_prediction(train_x, s_new_not_rest_012 ,xgboost, 100, 2, '신규방문자')
train_x, xgboost = xgboost_forcasting(t_page_not_rest_012, 100)
xgb_prediction(train_x, s_page_not_rest_012 ,xgboost, 100, 2, '페이지뷰')


train_x, xgboost = xgboost_forcasting(t_user_not_rest_34567, 200)
xgb_prediction(train_x, s_user_not_rest_34567 ,xgboost, 200, 19, '사용자')
train_x, xgboost = xgboost_forcasting(t_sess_not_rest_34567, 200)
xgb_prediction(train_x, s_sess_not_rest_34567 ,xgboost, 200, 19, '세션')
train_x, xgboost = xgboost_forcasting(t_new_not_rest_34567, 200)
xgb_prediction(train_x, s_new_not_rest_34567 ,xgboost, 200, 19, '신규방문자')
train_x, xgboost = xgboost_forcasting(t_page_not_rest_34567, 200)
xgb_prediction(train_x, s_page_not_rest_34567 ,xgboost, 200, 19, '페이지뷰')
df1 = pd.concat([s_user_rest_012, s_sess_rest_012.iloc[:,1], s_new_rest_012.iloc[:,1], s_page_rest_012.iloc[:,1]], axis = 1)
df2 = pd.concat([s_user_rest_34567, s_sess_rest_34567.iloc[:,1], s_new_rest_34567.iloc[:,1], s_page_rest_34567.iloc[:,1]], axis = 1)
df3 = pd.concat([s_user_not_rest_012, s_sess_not_rest_012.iloc[:,1], s_new_not_rest_012.iloc[:,1], s_page_not_rest_012.iloc[:,1]], axis = 1)
df4 = pd.concat([s_user_not_rest_34567, s_sess_not_rest_34567.iloc[:,1], s_new_not_rest_34567.iloc[:,1], s_page_not_rest_34567.iloc[:,1]], axis = 1)
## 예측값 합치기
for i in [df1, df2, df3, df4]:
    i.set_index('DateTime', inplace = True)
    
xgb_result = pd.concat([df1, df2, df3, df4]).sort_index()
print(xgb_result.shape)
xgb_result
(31, 4)
```

|            | 사용자 | 세션 | 신규방문자 | 페이지뷰 |
| ---------: | -----: | ---: | ---------: | -------: |
|   DateTime |        |      |            |          |
| 2020-12-09 |   3033 | 2990 |        771 |    68857 |
| 2020-12-10 |   3201 | 3410 |        877 |    71279 |
| 2020-12-11 |   3285 | 3608 |        851 |    75765 |
| 2020-12-12 |   2119 | 2077 |        460 |    46913 |
| 2020-12-13 |   2476 | 2360 |        418 |    58464 |
| 2020-12-14 |   3288 | 3881 |        822 |    83626 |
| 2020-12-15 |   3222 | 3781 |        818 |    84770 |
| 2020-12-16 |   3372 | 3686 |        801 |    85972 |
| 2020-12-17 |   2994 | 3253 |        813 |    82361 |
| 2020-12-18 |   3071 | 3090 |        798 |    82023 |
| 2020-12-19 |   2385 | 1946 |        434 |    49086 |
| 2020-12-20 |   2347 | 2147 |        410 |    48206 |
| 2020-12-21 |   3137 | 3662 |        810 |    80841 |
| 2020-12-22 |   3295 | 4003 |        770 |    78683 |
| 2020-12-23 |   3504 | 3561 |        790 |    79286 |
| 2020-12-24 |   3628 | 3470 |        743 |    82251 |
| 2020-12-25 |   2040 | 2104 |        426 |    51025 |
| 2020-12-26 |   2289 | 2527 |        451 |    52164 |
| 2020-12-27 |   2321 | 2075 |        574 |    50361 |
| 2020-12-28 |   3527 | 3652 |        780 |    80066 |
| 2020-12-29 |   3394 | 3772 |        816 |    79377 |
| 2020-12-30 |   3337 | 3584 |        797 |    82746 |
| 2020-12-31 |   2162 | 1969 |        480 |    48622 |
| 2021-01-01 |   2050 | 2054 |        525 |    50209 |
| 2021-01-02 |   1484 | 1487 |        313 |    28585 |
| 2021-01-03 |   1476 | 1417 |        328 |    30458 |
| 2021-01-04 |   2360 | 2284 |        592 |    46324 |
| 2021-01-05 |   2398 | 2088 |        515 |    46328 |
| 2021-01-06 |   3599 | 3468 |        755 |    91580 |
| 2021-01-07 |   3257 | 3415 |        737 |    91743 |
| 2021-01-08 |   3690 | 3326 |        738 |    87418 |

# 6. EXTRA TREE

```python
trn = pd.merge(train_data, add_data, how = 'inner', on = 'date')
trn.tail()
```

|      |       date | 사용자 | 세션 | 신규방문자 | 페이지뷰 | active_made | active_login_user | all_login_user | active_sub_user | active_sub_team | all_sub_user | all_sub_team |
| ---: | ---------: | -----: | ---: | ---------: | -------: | ----------: | ----------------: | -------------: | --------------: | --------------: | -----------: | -----------: |
|  721 | 2020-12-04 |   3189 | 3068 |        743 |    75730 |          42 |               227 |            276 |             136 |             136 |          352 |          352 |
|  722 | 2020-12-05 |   2055 | 2019 |        497 |    47638 |          31 |               129 |            143 |              70 |              74 |          197 |          197 |
|  723 | 2020-12-06 |   2119 | 2077 |        460 |    46914 |          24 |               147 |            169 |              72 |              73 |          192 |          192 |
|  724 | 2020-12-07 |   2979 | 2988 |        753 |    77443 |          70 |               279 |            345 |             113 |             112 |          236 |          236 |
|  725 | 2020-12-08 |   3033 | 2990 |        772 |    68857 |          70 |               265 |            299 |              98 |             100 |          223 |          223 |

#### 데이터 분할

- 이상점을 가지는 날들을 판별하기 위해 extra 알고리즘을 사용
- 19년도 까지는 이상점을 가지는 날이 없을 뿐더러 전체적으로 Y값들이 낮기에 최근 년도인 20년도 데이터만 사용

```python
## 훈련 데이터
trn_x = trn[(trn.date >= date(2020,1,1)) <= date(2020,12,8))].iloc[:,5:]
trn_y_user = trn[(trn.date >= date(2020,1,1)) <= date(2020,12,8))].reset_index(drop = True).iloc[:,1]
trn_y_sess = trn[(trn.date >= date(2020,1,1)) <= date(2020,12,8))].reset_index(drop = True).iloc[:,2]
trn_y_new = trn[(trn.date >= date(2020,1,1)) <= date(2020,12,8))].reset_index(drop = True).iloc[:,3]
trn_y_page = trn[(trn.date >= date(2020,1,1)) <= date(2020,12,8))].reset_index(drop = True).iloc[:,4]

## 테스트 데이터
test_x = add_data[(add_data.date > date(2020,12,8)) < date(2021, 1, 9))].reset_index(drop = True).iloc[:,1:]
print(trn_x.shape)
print(trn_y_user.shape)
print(test_x.shape)
(343, 7)
(343,)
(31, 7)
## 독립변수 확인
trn_x.head()
```

|      | active_made | active_login_user | all_login_user | active_sub_user | active_sub_team | all_sub_user | all_sub_team |
| ---: | ----------: | ----------------: | -------------: | --------------: | --------------: | -----------: | -----------: |
|  383 |          38 |               115 |            169 |              65 |              65 |          174 |          174 |
|  384 |          97 |               248 |            362 |              84 |              84 |          210 |          210 |
|  385 |          63 |               179 |            262 |              88 |              88 |          210 |          210 |
|  386 |          61 |               146 |            206 |              86 |              87 |          191 |          191 |
|  387 |          50 |               131 |            174 |              94 |              91 |          232 |          232 |

```python
### Grid Search CV를 통해 발견한 best parameter 설정
## 베스트 파라미터 저장
final_grid1_params = {'max_depth': 9, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 600, 'random_state': 40}
final_grid2_params = {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 700, 'random_state': 40}
final_grid3_params = {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 550, 'random_state': 40}
final_grid4_params = {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 400, 'random_state': 40}

## best 파라미터 입력
final_grid1 = ExtraTreesRegressor(**final_grid1_params)
final_grid2 = ExtraTreesRegressor(**final_grid2_params)
final_grid3 = ExtraTreesRegressor(**final_grid3_params)
final_grid4 = ExtraTreesRegressor(**final_grid4_params)

## 피팅
final_grid1.fit(trn_x, trn_y_user)
final_grid2.fit(trn_x, trn_y_sess)
final_grid3.fit(trn_x, trn_y_new)
final_grid4.fit(trn_x, trn_y_page)

## 예측
pred_user_grid = pd.Series(final_grid1.predict(test_x))
pred_sess_grid = pd.Series(final_grid2.predict(test_x))
pred_new_grid = pd.Series(final_grid3.predict(test_x))
pred_page_grid = pd.Series(final_grid4.predict(test_x))
## 이후 날짜도 합치기
extra_result = pd.concat([submission.DateTime.reset_index(drop=True), pred_user_grid, pred_sess_grid, pred_new_grid, pred_page_grid], axis = 1)
extra_result.set_index('DateTime', inplace = True)
extra_result.rename(columns = {0:'사용자',1:'세션',2:'신규방문자',3:'페이지뷰'}, inplace = True)
extra_result = extra_result.round().astype(int)
extra_result
```

|            | 사용자 | 세션 | 신규방문자 | 페이지뷰 |
| ---------: | -----: | ---: | ---------: | -------: |
|   DateTime |        |      |            |          |
| 2020-12-09 |   3895 | 3887 |       1015 |   103843 |
| 2020-12-10 |   2906 | 2972 |        761 |    58831 |
| 2020-12-11 |   2754 | 2693 |        698 |    67344 |
| 2020-12-12 |   2160 | 2093 |        468 |    52298 |
| 2020-12-13 |   1917 | 1854 |        414 |    45641 |
| 2020-12-14 |   3067 | 3005 |        756 |    77360 |
| 2020-12-15 |   3332 | 3280 |        849 |    81826 |
| 2020-12-16 |   2934 | 2897 |        791 |    70723 |
| 2020-12-17 |   3032 | 3002 |        758 |    74137 |
| 2020-12-18 |   3938 | 3946 |       1033 |   105203 |
| 2020-12-19 |   2529 | 2485 |        646 |    63152 |
| 2020-12-20 |   2754 | 2727 |        690 |    70866 |
| 2020-12-21 |   3804 | 3775 |        941 |    95117 |
| 2020-12-22 |   3872 | 3779 |       1007 |    96649 |
| 2020-12-23 |   3267 | 3203 |        790 |    87515 |
| 2020-12-24 |   3231 | 3162 |        805 |    86860 |
| 2020-12-25 |   2481 | 2412 |        577 |    69352 |
| 2020-12-26 |   3263 | 3189 |        826 |    83881 |
| 2020-12-27 |   4145 | 4256 |       1157 |   115525 |
| 2020-12-28 |   4180 | 4201 |       1152 |   112500 |
| 2020-12-29 |   3861 | 3799 |        984 |    97383 |
| 2020-12-30 |   4829 | 4863 |       1348 |   135071 |
| 2020-12-31 |   4924 | 5237 |       1657 |   149325 |
| 2021-01-01 |   2785 | 3004 |        753 |    50618 |
| 2021-01-02 |   3188 | 3217 |        828 |    75544 |
| 2021-01-03 |   3144 | 3112 |        859 |    81096 |
| 2021-01-04 |   4034 | 4044 |       1089 |   108745 |
| 2021-01-05 |   3968 | 3942 |        994 |   107817 |
| 2021-01-06 |   4766 | 4923 |       1401 |   141304 |
| 2021-01-07 |   4377 | 4417 |       1203 |   117112 |
| 2021-01-08 |   3956 | 3913 |        961 |   108118 |

# 7. Weighted average ensemble

- 알고리즘 결과값들을 비교한 후 각기 다르게 가중치 부여

```python
### DateTime 인덱스 설정
lstm_result.index = pd.to_datetime(lstm_result.index)
xgb_result.index = pd.to_datetime(xgb_result.index)
extra_result.index = pd.to_datetime(extra_result.index)

### 결과값 비교
fig = make_subplots(
    subplot_titles=("사용자","세션","신규방문자","페이지뷰"),
    horizontal_spacing=0.1,
    rows=2, cols=2 )

fig.add_trace(go.Scatter(x=lstm_result.index, y=lstm_result['사용자'],name="lstm_result",
                          marker=dict(size=9, color='Red')), row=1, col=1)
fig.add_trace(go.Scatter(x=xgb_result.index, y=xgb_result['사용자'],name="xgb_result",
                         marker=dict(size=9, color='Green')), row=1, col=1)
fig.add_trace(go.Scatter(x=extra_result.index, y=extra_result['사용자'],name="extra_result",
                         marker=dict(size=9, color='Blue')), row=1, col=1)

fig.add_trace(go.Scatter(x=lstm_result.index, y=lstm_result['세션'],name="lstm_result",
                          marker=dict(size=9, color='Red'),showlegend = False), row=1, col=2)
fig.add_trace(go.Scatter(x=xgb_result.index, y=xgb_result['세션'],name="xgb_result",
                         marker=dict(size=9, color='Green'),showlegend = False), row=1, col=2)
fig.add_trace(go.Scatter(x=extra_result.index, y=extra_result['세션'],name="extra_result",
                         marker=dict(size=9, color='Blue'),showlegend = False), row=1, col=2)

fig.add_trace(go.Scatter(x=lstm_result.index, y=lstm_result['신규방문자'],name="lstm_result",
                          marker=dict(size=9, color='Red'),showlegend = False), row=2, col=1)
fig.add_trace(go.Scatter(x=xgb_result.index, y=xgb_result['신규방문자'],name="xgb_result",
                         marker=dict(size=9, color='Green'),showlegend = False), row=2, col=1)
fig.add_trace(go.Scatter(x=extra_result.index, y=extra_result['신규방문자'],name="extra_result",
                         marker=dict(size=9, color='Blue'),showlegend = False), row=2, col=1)

fig.add_trace(go.Scatter(x=lstm_result.index, y=lstm_result['페이지뷰'],name="lstm_result",
                          marker=dict(size=9, color='Red'),showlegend = False), row=2, col=2)
fig.add_trace(go.Scatter(x=xgb_result.index, y=xgb_result['페이지뷰'],name="xgb_result",
                         marker=dict(size=9, color='Green'),showlegend = False), row=2, col=2)
fig.add_trace(go.Scatter(x=extra_result.index, y=extra_result['페이지뷰'],name="extra_result",
                         marker=dict(size=9, color='Blue'),showlegend = False), row=2, col=2)

fig.update_layout(height =800, width =950, title_text="Scatter Plot for result", template='ggplot2')
fig.show()
```

Dec 132020Dec 20Dec 27Jan 320212000300040005000Dec 132020Dec 20Dec 27Jan 320212000300040005000Dec 132020Dec 20Dec 27Jan 3202150010001500Dec 132020Dec 20Dec 27Jan 3202140k60k80k100k120k140k

lstm_resultxgb_resultextra_resultScatter Plot for result사용자세션신규방문자페이지뷰













```python
final_result = ((lstm_result*0.3)+(xgb_result*0.7))*0.2 + extra_result*0.8
final_result = final_result.astype(int)
final_result
```

|            | 사용자 | 세션 | 신규방문자 | 페이지뷰 |
| ---------: | -----: | ---: | ---------: | -------: |
|   DateTime |        |      |            |          |
| 2020-12-09 |   3671 | 3657 |        949 |    95930 |
| 2020-12-10 |   2956 | 3037 |        777 |    61356 |
| 2020-12-11 |   2884 | 2873 |        734 |    70543 |
| 2020-12-12 |   2150 | 2086 |        467 |    51386 |
| 2020-12-13 |   2000 | 1930 |        415 |    47526 |
| 2020-12-14 |   3116 | 3146 |        768 |    78957 |
| 2020-12-15 |   3292 | 3325 |        835 |    81971 |
| 2020-12-16 |   3025 | 3036 |        793 |    74117 |
| 2020-12-17 |   3043 | 3052 |        767 |    75651 |
| 2020-12-18 |   3769 | 3778 |        985 |   100171 |
| 2020-12-19 |   2479 | 2384 |        610 |    60098 |
| 2020-12-20 |   2658 | 2611 |        641 |    66474 |
| 2020-12-21 |   3704 | 3757 |        928 |    93662 |
| 2020-12-22 |   3716 | 3740 |        949 |    92311 |
| 2020-12-23 |   3258 | 3216 |        780 |    84799 |
| 2020-12-24 |   3270 | 3196 |        797 |    85103 |
| 2020-12-25 |   2448 | 2404 |        571 |    67240 |
| 2020-12-26 |   3024 | 3000 |        750 |    76407 |
| 2020-12-27 |   3728 | 3786 |       1030 |   101449 |
| 2020-12-28 |   4006 | 4046 |       1077 |   105633 |
| 2020-12-29 |   3713 | 3720 |        942 |    93073 |
| 2020-12-30 |   4454 | 4523 |       1224 |   122943 |
| 2020-12-31 |   4393 | 4627 |       1436 |   130387 |
| 2021-01-01 |   2638 | 2823 |        713 |    50904 |
| 2021-01-02 |   2875 | 2907 |        740 |    67761 |
| 2021-01-03 |   2848 | 2820 |        763 |    72850 |
| 2021-01-04 |   3739 | 3743 |        994 |    98859 |
| 2021-01-05 |   3673 | 3612 |        899 |    97362 |
| 2021-01-06 |   4515 | 4627 |       1267 |   131409 |
| 2021-01-07 |   4159 | 4216 |       1110 |   111612 |
| 2021-01-08 |   3848 | 3764 |        911 |   102452 |

```python
submission = pd.read_csv('data/submission.csv', encoding = 'euc-kr')
submission.DateTime = pd.to_datetime(submission.DateTime).dt.date

for i in range(31):
    submission['사용자'][i+30] = final_result['사용자'][i]
    submission['세션'][i+30] = final_result['세션'][i]
    submission['신규방문자'][i+30] = final_result['신규방문자'][i]
    submission['페이지뷰'][i+30] = final_result['페이지뷰'][i]

submission
```

|      |   DateTime | 사용자 | 세션 | 신규방문자 | 페이지뷰 |
| ---: | ---------: | -----: | ---: | ---------: | -------: |
|    0 | 2020-11-09 |      0 |    0 |          0 |        0 |
|    1 | 2020-11-10 |      0 |    0 |          0 |        0 |
|    2 | 2020-11-11 |      0 |    0 |          0 |        0 |
|    3 | 2020-11-12 |      0 |    0 |          0 |        0 |
|    4 | 2020-11-13 |      0 |    0 |          0 |        0 |
|  ... |        ... |    ... |  ... |        ... |      ... |
|   56 | 2021-01-04 |   3739 | 3743 |        994 |    98859 |
|   57 | 2021-01-05 |   3673 | 3612 |        899 |    97362 |
|   58 | 2021-01-06 |   4515 | 4627 |       1267 |   131409 |
|   59 | 2021-01-07 |   4159 | 4216 |       1110 |   111612 |
|   60 | 2021-01-08 |   3848 | 3764 |        911 |   102452 |

61 rows × 5 columns

```python
submission.to_csv('submission.csv', index = False, encoding = 'euc-kr')
```