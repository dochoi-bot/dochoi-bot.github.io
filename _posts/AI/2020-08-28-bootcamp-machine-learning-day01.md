---
title: Bootcamp Machine Learning day01
layout: post
category: ai
tags:
- AI
- MachineLearning
excerpt: 머신러닝 배우기 두번째 걸음
comments: true
---

# Bootcamp Machine Learning day01

- 머신러닝 기초공부 day01에서 느낀점을 적는다.


- [Bootcamp Machine Learning day01](#bootcamp-machine-learning-day01)
  - [ex00](#ex00)
  - [ex01](#ex01)
  - [ex02](#ex02)
  - [ex03](#ex03)
  - [Linear Gradient Iterative Version](#linear-gradient-iterative-version)
  - [ex04](#ex04)
  - [**Linear Gradient Vectorized Version**](#linear-gradient-vectorized-version)
  - [ex05](#ex05)
  - [Gradient Descent](#gradient-descent)
        - [몇가지 주의사항](#몇가지-주의사항)
    - [목표](#목표)
  - [ex06](#ex06)
  - [Linear Regression](#linear-regression)
  - [ex07](#ex07)
    - [목표 :](#목표-)
    - [설계](#설계)
  - [ex08](#ex08)
    - [정규화](#정규화)
  - [ex09](#ex09)
    - [목표](#목표-1)
  - [ex10](#ex10)


## ex00

day00의

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg4.png)

를 그대로 구현하는것이다.

## ex01

day00의 cost함수를 그대로 구현하는것이다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg2.png)

## ex02

day00의 마지막 문제이다



## ex03



문제에 앞서 간단한 이론이 나온다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg6.png)

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg5.png)

왼쪽 그래프는 오른쪽에서 theta_1의 값이 노란색 점일때 예측과 실제값이 얼마나 유사한지 보여준다.
cost값이 0일때가 이상적인 모델이므로 오른쪽 그래프에서 기울기가 0일때,  내가 원하는 theta값이다.이를 찾아가는 간단한 알고리즘이다. 기울기가 양수이면 theta_1의 값을 줄이고 기울기가 음수이면 theta_1의 값을 늘려서 이상적인 값을 찾는다.

하지만 우리가 cost를 계산할때는 theta_0값도 존재한다. 즉 파라미터가 2개이다.
따라서 theta_0에따른 cost함수에 대한 기울기도 구해야한다.
즉, 차원이 2 x 1 인 벡터를  구할것이다.

## Linear Gradient Iterative Version

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg7.png)

cost 함수의 기울기를 구하는 문제이다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg8.png)

여기서 y_hat(i)에 아래식을 넣고 theta_0과 theta_1에대해 미분하면 위에 두가지 식이 유도된다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg9.png)

예제에 오류가 있다.

```python
x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]) 		y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])

# Example 0:
theta1 = np.array([2, 0.7])
simple_gradient(x, y, theta1)
# Output: array([21.0342574, 587.36875564])

# Example 1:
theta2 = np.array([1, -0.4])
simple_gradient(x, y, theta2)
# Output: array([58.86823748, 2229.72297889])
```

```
[21.0342574, 587.36875564]
[58.86823748, 2229.72297889]
```

가아닌

```
[-19.034257402, -586.668755635948]
[-57.868237476000004, -2230.1229788875144]
```

값이 출력되어야한다.

https://github.com/42-AI/bootcamp_machine-learning/issues/64

이곳에 issues가 올라와있지만 아직 해결되지 않은듯 싶다.



## ex04

## **Linear Gradient Vectorized Version**

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg10.png)

목표는 ex03과 동일하지만 벡터곱을 통해서 결과를 도출해보자

저 식의 유도 과정에 대해 알아보자

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg11.png)

여기서 x_0는 1이다.

즉, X'에 값이 1인 열을 추가하면(첫번째 열로)

기울기를 구하는 식이 하나의 식으로 통합된다. 이로써 데이터셋(x의 행의 수)이 많더라도 코드는 한줄로 작성이 가능하다.

<img src="https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg14.jpg" alt="MLimg4" style="zoom: 50%;" />
여기서 모든 데이터셋에대한 결과를 출력하려면 식을 변형한다.



<img src="https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg13.jpg" alt="MLimg4" style="zoom: 50%;" />

<img src="https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg12.jpg" alt="MLimg4" style="zoom: 33%;" />

y배열은 y_hat과 비교할 실제 결과값이다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg10.png)

따라서 이 식이 유도된다.

numpy에서 transpose함수 3가지이다

a가 numpy 배열이라고 했을때 ,

- a.T

- numpy.transpose(a)

- numpy.swapaxex(a,0,1)

  나는 가장 간단한 a.T를 이용하여 구현하였다.



## ex05

## Gradient Descent

경사하강법이다. 머신러닝 공부하기 전에도 어디선가 많이 들어봤다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg12.png)

간단하다 이전에 말했듯이 cost를 0으로 만들기위한 세타값을 찾아가는 과정이다.

기울기가 음수면 세타를 증가하고 기울기가 양수면 세타를 감소시킨다.

##### 몇가지 주의사항

- α라는 learning rate를 두어 minium값을 널뛰기 하지않게 막아준다.

- 수렴할지 여부는 모르니 loop의 수를 정한다.

- alpha와 cycle수는 trial and error기법으로 시행착오를 통해서 알아내야한다..



문제에 들어가보자

derivatives함수 사용을 금지해놨다. 미분해서 최솟값으로 간단히 구하는걸 막아논듯 싶다.

###  목표

-  Be able to explain what it means to ﬁt a Machine Learning model to a dataset

-  Implement a function that performs Linear Gradient Descent (LGD).

**데이터셋을 머신러닝 모델에 fit한다는 의미를 이해하고** **LGD를 구현해보자**

이전에 구한 cost함수를 적용해주기만 하면 된다.

인덱스와 값을 동시에 반환하는 for문 함수 enumerate를 이용해봤다

```python
while max_iter:
    for i, v in enumerate(gradient(x, y, theta)):
        theta[i] -= v
        max_iter -= 1
```

이 문제에서 갑자기 x, y shape가 (n,)에서 (n, 1)로 주어진다.
이 때문에 차원을 맞추기 위해 함수를 다시 구현하였다 ..

```py
b.reshape(-1,1)
```

reshape에 -1을 넣어주면 자동으로 차원이 맞춰진다.



## ex06

## Linear Regression

선형 회귀,, 확률 및 통계에서 배운적 있다.

종속 변수 y와 한 개 이상의 독립 변수 X와의 선형 상관 관계를 모델링한다.

sklearn에 있는 LinearRegression 클래스를 구현해본다.

계산할 때 차원이 (m,)이면 차원을 (m, 1)로 확장시켜서 계산해준다.

https://github.com/42-AI/bootcamp_machine-learning/issues/66

example1.0의 예제가 잘못되었다..

저 결과가 나오기 위해선 alpha와 max_iter를 5e-8, 1500000으로 해주어야 한다.

예제처럼  기본 alpha값과 max_iter가 0.001, 1000일경우

theta는

[[5.13756398]
 [1.02736452]]

이나와 예제 출력이 모두 바뀌게 된다.

아마 출제자의 실수가 아니라면 의도는
무수한 시도 끝에 적절한 alpha와 max_iter를 찾으라는 뜻인거 같다.



## ex07

### 목표 :

- 선형 회귀 분석 모델을 평가한다. (hypothesis funchtion이 주어지고, 데이터셋이 적은 상황에서)

- cost function을 조작하여 plot하고, 간단히 plot을 분석한다



### 설계

resources 폴더에 있는 are_blue_pills_magics.csv를 분석해야하기 때문에

여기있는 데이터를 읽어 모델을 적용시키자

가설공식

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg13.png)

이전과 같다.

2가지 그래프를 그려야 한다.

- 실제 관측값을 점으로 찍고 내가 예측한 선을 긋는다. (최선의 예측)
  - x축 :  quantity of "blue pills"
  - y축 : spacecraft piloting score

- theta의 변화에 따른 cost function의 그래프를 그려야한다.

**cost function**은

이전에 구현해본 **MSE**(Mean Squared Error)를 이용할것이다.

 ex06에서 구현한 클래스로 그래프를 그려보았다.

```python
x = Xpill
y = Yscore
plt.scatter(x, y)
linear_model1.fit_(x, y)
plt.xlabel('Quantity of blue pill (in micrograms)')
plt.ylabel('Space driving score')
plt.title('simple plot')
plt.plot(x, linear_model1.predict_(x), color='green')
plt.legend(['S_true', 'S_predict'])
plt.show()
```



구현하던중 특이한걸 발견했다.

python에서 클래스 안의 변수를 접근할때

model1.thetas = 1하면 대입이 되지만

x = model1.thetas를 하면 주솟값이 복사되어

x의 값이 바뀌면 mode1.thetas의 값이 바뀐다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg15.png)

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg14.png)



그래프 그리는것을 성공하였다 문제에 주어진 조건이 너무 부실하여 구현 하는데 조금 오래걸렸다.

```python
plt.ylim([20, 140])
```

이 함수로 y축값을 조정해주고 theta0변동 폭을 줄이니 ppt와 유사한 함수가 나왔다.

```python
x= Xpill
y = Yscore
plt.scatter(x, y)
linear_model1.fit_(x, y)
plt.xlabel('Quantity of blue pill (in micrograms)')
plt.ylabel('Space driving score')
plt.title('simple plot')
plt.plot(x, linear_model1.predict_(x), color='green')
plt.legend(['S_true', 'S_predict'])
plt.show()




legends = []
plt.xlabel('theta1')
plt.ylabel('cost func J(theta0, theta1)')
theta0s = np.arange(linear_model1.thetas[0] - 15, linear_model1.thetas[0] + 15, 5)
theta1 = linear_model1.thetas[1].copy()
print(linear_model1.thetas)
for theta0 in theta0s:
    linear_model1.thetas[0]= theta0
    linear_model1.thetas[1] = theta1
    legends.append('theta0 = ' + str(theta0))
    temp = np.arange(theta1 - 5 , theta1 + 5, 0.1)
    # print(linear_model1.thetas)
    temp2 = []
    for i in temp :
        linear_model1.thetas[1] = i
        temp2.append(linear_model1.cost_(x,y))
    plt.plot(temp, temp2)
    print(min(temp2))
plt.legend(legends)
plt.ylim([20, 140])
plt.show()
```

전체 코드이다.



## ex08

질문 시간이다.

**1 - What is a hypothesis and what is its goal?**
(It’s a second chance for you to say something intelligible, no need to thank us!)

- 말 그대로 가설이다. 독립된 x값와 종속된 y값이 주어진 파라미터(theta0, theta1) 등에 어떻게 변화 할지를 예측하여 수식화한다. goal은 실제 결과값과 같게 모델을 만드는 것이다.

**2 - What is the cost function and what does it represent?**

- 평가함수이다. 내가 만든 모델과 실제 결과값의 차이를 반환 해주는 함수이다. 여기서 이 cost를 계산하는 방식은 여러가지가 있는데 경우에 따라 다르지만 MSE(표준 제곱 오차)를 많이 사용한다.

**3- What is Linear Gradient Descent and what does it do?**
(hint: you have to talk about J, its gradient and the theta parameters...)

- cost함수의 gradient를 계산하면 cost를 가장 작게 만드는 theta parameters를 알 수 있다.

  이로써 예측 모델의 parameters를 계산할 수 있다.

**4 - What happens if you choose a learning rate that is too large?**

- 함수가 발산하여 nan이 출력된다. 또한, 올바른 theta parameters를 jump해버린다.

**5 - What happens if you choose a very small learning rate, but still a suﬃcient number of cycles?**

- 초기에 실제 모델과 꽤 유사하게 결과를 만드는(cost function이 최소에 가까운) theta parameters를 넣었다면 문제가 안되지만 반대의 경우 cost를 가장 작게 만드는 theta parameters에 다가갈수 없다. 이때는 cycles를 늘려주어야 한다.

**6 - Can you explain MSE and what it measures**

- 오차의 제곱에 평균을 취한것이다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg3.png)

여기서 보이는 빨간 점선들의 거리의 평균을 취한것이다.



### 정규화

Normalization 정규화이다.
x vector의 크기는 다양하다. 어쩔때는 굉장히 클 수도 있고, 작을 수도 있다.
두경우 모두 Gradient Descent수렴을 늦출 수 있다. 이를 피하기 위해서 크기를 정규화한다.
학부생 컴퓨터비전 시간에 **데이터가 너무 크면 컴퓨터에서 계산하는 과정에서 오차가 크게 발생**한다고 했던 교수님의 말씀이 생각난다. 이를 줄여주기 위해 정규화했던것이 기억난다.
데이터를 [0,1] 이나 [-1, 1]로 매핑해본다.

## ex09



 Z_score 정규화

### 목표

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg16.png)

즉 x의 원소에 평균을 빼고 표준편차로 나눠준다.

모분산 모평균, 표본평균, 표본분산의 개념이 헷갈린다.

https://blog.naver.com/dalsapcho/20147545698

표준편차에서 m이 아닌 m-1로 나눠주는 이유는 모분산을 추정할때 m-1로 나눠줘야 정확한 값을 얻을 수있기 때문이다. 이는 수학적인 과정이라 아래에 나타낸다.

https://blog.naver.com/physicopianist/221079231724

위 공식은 정규화 공식이다.  그렇다면 이는 원래 x가 표준 정규 분포를 따른다고 가정하고 변환하는것이다. 우리가 가지고있는 데이터는 어디까지나 표본이기 때문에 표본표준편차공식을 이용해야한다.(m과 m-1의 차이)



너무 깊게 들어가지 말고 이정도까지만 공부하고 식을 적용해보자.
example에서는 표준편차에서 m-1이아닌 m로 나눠주었다.
즉 example에선 m과 m-1차이를 대수롭게 생각하지 않은것 같다.



```python
import numpy as np
import math

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args: x:
    has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception. """
    if len(x) == 0:
        return none
    mu = sum(x)/len(x)
    temp = 0.0
    for elem in x:
        temp += ((elem - mu) * (elem - mu))
    var = temp
    std = math.sqrt(var/ (len(x) - 1))
    return (x - mu) / std
```



## ex10

Min-max Standardization

이는 ex09보다 훨씬 직관적이다.

![MLimg4](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Machine%20Learning/image/MLimg17.png)

이는 0부터 1까지의 값으로 매핑된다.

```python
def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.,
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns: x' as a numpy.ndarray. None if x is a non-empty numpy.ndarray.
    Raises: This function shouldn't raise any Exception. """
    pivot = max(x) - min(x)
    return (x - min(x)) / pivot
```

이로써 마지막 문제가 끝났다.

[깃허브 링크](https://github.com/dochoi-bot/Bootcamp_Machine_Learning)