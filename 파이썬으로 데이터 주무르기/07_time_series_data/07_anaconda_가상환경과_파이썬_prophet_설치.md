# 7장 anaconda 가상환경 & prophet 설치

### 1. anaconda 설치

anaconda를 설치한다.



### 2. anaconda 가상환경 만들기

#### 2-1. anaconda prompt 실행

jupyter notebook을 작성할 임의의 폴더에서 anaconda prompt를 실행한다.

#### 2-2. python 3.8.13 버전으로 가상환경 만들기

```bash
$ conda create -n 가상환경이름 python=3.8.13
# 따로 폴더가 생성되지는 않는다
```

#### 2-3. 가상환경 activate

```bash
$ conda activate 가상환경이름
```

#### 2-4. 책에 나와 있는 모듈을 가상환경에 모두 설치해준다



### 3. 코드 작성해보기

#### 3-1. prophet 관련 에러 발생

AttributeError: 'Prophet' object has no attribute 'stan_backend'



### 4. 에러 해결하기 (링크 참고)

#### 4-1. pystan 2.19.1.1 버전으로 먼저 설치하기(중요)

https://github.com/facebook/prophet/issues/1574

```bash
pip install pystan==2.19.1.1
```



#### 4-2. prophet 설치하기

https://joytk.tistory.com/73

```bash
# pip install로 안되기 때문에
conda install -c conda-forge fbprophet
conda install -c plotly plotly
```



#### 4-3. jupyter notebook이 정상적으로 실행되지 않는 경우

```bash
pip install zipp
```