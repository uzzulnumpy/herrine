# Lv5_EDA_데이터 불러오기 및 확인



### 데이터를 n번째 행까지만 불러오기

```python
# 예) 데이터를 3번째 행까지만 불러오기

import pandas as pd

train_3 = pd.read_csv('data/train.csv', nrows=3)
```



### 데이터의 n번째 행을 컬럼으로 지정하여 불러오기

```python
# 예) 데이터의 2번째 행을 컬럼으로 지정
import pandas as pd

train = pd.read_csv('data/train.csv',header = 1) 
test = pd.read_csv('data/test.csv',header = 1)
```



###  원하는 컬럼을 index로 지정하여 불러오기

```python
train_index = pd.read_csv('data/train.csv',index_col = 'index') 
test_index = pd.read_csv('data/test.csv',index_col = 'index')
```

![image-20220507215302976](Lv5_EDA_%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%88%EB%9F%AC%EC%98%A4%EA%B8%B0%20%EB%B0%8F%20%ED%99%95%EC%9D%B8.assets/image-20220507215302976.png)



### 데이터에서 결측치를 제외하고 불러오기

```python
# na_filter=False 옵션으로 결측치를 제외한다.
# 데이터를 결측치를 제외하고 불러오기
import pandas as pd

train_notnull = pd.read_csv('data/train.csv',na_filter =False)
test_notnull = pd.read_csv('data/test.csv',na_filter =False)

# 확인
train_notnull.isnull().sum()
```



### 데이터에서 뒤에서 n개의 행 제외하고 불러오기

```python
import pandas as pd


train_skipfooter = pd.read_csv('data/train.csv' , skipfooter=5 )
test_skipfooter = pd.read_csv('data/test.csv' , skipfooter=5 )
```



### 데이터의 인코딩 형식을 맞춰서 불러오기

'내가 불러오고자 하는 데이터의 encoding과 python encoding의 설정이 맞지 않는 경우'

```python
import pandas as pd

train = pd.read_csv('train.csv',encoding = 'cp949')
```



### 데이터를 불러올 때 컬럼명을 지정해서 불러오기

```python
import pandas as pd


train_names = pd.read_csv('data/train.csv',names=['인덱스','카테고리','텍스트'])
test_names = pd.read_csv('data/test.csv',names=['인덱스','카테고리','텍스트'])
```

![image-20220507225405518](Lv5_EDA_%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%88%EB%9F%AC%EC%98%A4%EA%B8%B0%20%EB%B0%8F%20%ED%99%95%EC%9D%B8.assets/image-20220507225405518.png)



### index는 제외하고 데이터 저장하기

'dataframe을 to_csv 메서드로 저장할 때 인덱스도 데이터에 포함되어 저장되고 인덱스가 추가로 생긴다. 따라서 원래 인덱스는 제외하고 저장해야 한다. '

![img](data:image/png;base64,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)

```python
import pandas as pd

train.to_csv('data/train.csv',index = False) 
test.to_csv('data/test.csv',index = False)
```



### value_counts(): 어떤 컬럼의 unique한 value들을 count

```python
# train 데이터에서 category 컬럼의 고유값의 개수를 출력
train['category'].value_counts()
```

```python
2    13362
1    13337
0    13301
Name: category, dtype: int64
```

