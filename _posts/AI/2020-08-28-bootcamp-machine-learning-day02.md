---
title: Bootcamp Machine Learning day02
layout: post
category: ai
tags:
- AI
- MachineLearning
excerpt: 머신러닝 배우기 세번째 걸음
comments: true
---

# Bootcamp Machine Learning day02

- 머신러닝 기초공부 day02에서 느낀점을 적는다.


- [Bootcamp Machine Learning day02](#bootcamp-machine-learning-day02)
  - [ex00](#ex00)
  - [ex01](#ex01)
  - [multivariate model](#multivariate-model)
  - [ex02](#ex02)
  - [ex03](#ex03)
  - [ex04](#ex04)
  - [ex05](#ex05)
  - [ex06](#ex06)
  - [ex07](#ex07)
  - [ex08](#ex08)
    - [Practicing Multivariate Linear Regression](#practicing-multivariate-linear-regression)
    - [목표](#목표)
    - [개요](#개요)
  - [Part One(Univariate Linear Regression)](#part-oneunivariate-linear-regression)
    - [Age 그래프 그리기](#age-그래프-그리기)
    - [Thrust power 그래프 그리기](#thrust-power-그래프-그리기)
    - [Total distance 그래프 그리기`](#total-distance-그래프-그리기)
  - [Part two(Multivariate Linear Regression)](#part-twomultivariate-linear-regression)
    - [목표](#목표-1)
  - [ex09](#ex09)
  - [Introducing Polynomial Models](#introducing-polynomial-models)
    - [**Polynomial Hypothesis**](#polynomial-hypothesis)
  - [ex10](#ex10)
  - [Polynomial models](#polynomial-models)
    - [목표](#목표-2)
  - [ex11](#ex11)
    - [목표](#목표-3)
  - [Plotting Curves With Matplotlib](#plotting-curves-with-matplotlib)
  - [ex12](#ex12)
    - [Let’s PLOT some Polynomial Models!](#lets-plot-some-polynomial-models)
  - [Lost in Overﬁtting](#lost-in-overﬁtting)
  - [ex13](#ex13)
    - [DataSpliter](#dataspliter)
    - [목표](#목표-4)
  - [ex14](#ex14)
  - [ex15](#ex15)
    - [Question Time!](#question-time)


## ex00

day01 의 Linear Regression 클래스 구현이다.



## ex01

day01의 ex08질문들이다.



## multivariate model

![MLimg0](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg0.png)

지금까지 hypothesis를 정말 간단한걸 사용했다.
이걸로는 복잡한 세상을 예측할 수 없다.
하나의 변수만으로는 복잡한 현상을 예측할 수 없다.

예를 들어 날씨를 예측할 때 온도뿐아니라 습도, 바람, 일조량등의 변수를 추가하여 계산한다면 더 정확하게 예측 할 수 있다.

따라서 오늘은 **multivariate model**에 대해 다뤄볼것이다.

## ex02

![MLimg0](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg18.png)

multivariate model을 다루기 위해 벡터가아닌 매트릭스를 이용한다.

M * N 으로 구성되어있으며 n 은 features(variables)의 index이고 M은 각각의 데이터셋이다.

```python
def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
     Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
      Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray. None if x or theta dimensions are not appropriate.
     Raises: This function should not raise any Exception.
    """
    if len(x) == 0:
        return None
    answer = []
    for i in range(len(x)):
        temp = theta[0]
        for j in range(len(x[0])):
            temp += (x[i][j] * theta[j + 1])
        answer.append(temp)
    return (answer)
```

반복문 버전으로 구현하였다.

## ex03



```python
def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
     Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
      Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray. None if x or theta dimensions are not appropriate.
     Raises: This function should not raise any Exception.
    """
    def add_intercept(x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)
    if len(x) == 0:
        return None
    return (add_intercept(x) @ theta)

```

값이 1인 칼럼을 추가하여 행렬곱하는 식으로 구현하였다.

## ex04

**multivariate model**에서의 cost function

이 함수에는 x나 theta가 필요하지 않고

y_hat과 y만 필요하기 때문에 day01에서 구현한 cost함수와 동일하게 사용할 수 있다.



## ex05

**multivariate model**에서의 Linear Gradient 구하기

![MLimg0](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg19.png)

여기선 또 theta의 수와 x의 원소 수를 같게 주었다...  즉 x_0이 주어지지 않았다..

일관되게 예제를 만들어 주면 좋겠다..





## ex06

 **Multivariate Gradient Descent**

미분함수 있는 라이브러리 금지

여기선 또 theta의 수가 더 많다 ..(x_0)이 주어졌다.

하지만 이전에 일변수 모델과 식은 유사하여 구현은 간단하였다.

gradient를 계산하고, n_cycle만큼 , alpha rate로보정하여 이상의 theta값을 구한다.

```python
import numpy as np

def fit_(x, y, theta, alpha, n_cycle):
    """ Description: Fits the model to the training dataset contained in x and y.
    Args:
    x:
    has to be a numpy.ndarray, a vector of dimension m * n:  (number of training examples, number of features).
    y:
    has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    Raises: This function should not raise any Exception. """
    def gradient(x, y, theta):
        def add_intercept(x):
            if len(x) == 0 or x.ndim >= 3:
                return None
            if x.ndim == 1:
                return np.vstack((np.ones(len(x)), x)).T
            else:
                return np.insert(x, 0, 1, axis=1)
        def predict_(x, theta):
            if x.ndim == 1:
                x = x[:,np.newaxis]
            if len(x) == 0:
                return None
            return add_intercept(x) @ theta
        if len(x) == 0 or len(y) == 0 or len(theta) == 0:
            return None
        return add_intercept(x).T @ (predict_(x, theta) - y) / len(x)


    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    theta_r = np.array(theta, dtype=float)
    while n_cycle:
        for i, v in enumerate(gradient(x, y, theta_r)):
            theta_r[i] -= (alpha * v)
        n_cycle -= 1
    return theta_r

```



## ex07

지금까지 했던것을 클래스로 만들어본다.

```python
X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])
print(mylr.cost_(X,Y))
mylr.fit_(X, Y)
print(mylr.thetas)
print(mylr.predict_(X))
print(mylr.cost_elem_(X,Y))
print(mylr.cost_(X,Y))
```

이 코드에 대한 결과값이다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg20.png" alt="MLimg0"  />

이정도로 alpha값 0.0005, n_cycle=322000으로 만족하겠다. (사이클을 늘리면 시간이 너무 많이걸린다.) cost function이 0.1이니 만족한다.



## ex08

### Practicing Multivariate Linear Regression



sklearn 금지, numpy, matplotlib만 사용이 허가된다.

### 목표

- dataset with multiple features를 regression model에 **Fit** 하기
- 예측을 그래프로 그리고, 이를 설명하라.

### 개요

csv파일을 읽어서 그래프를 plot하라



## Part One(Univariate Linear Regression)

- 일변량 선형 회귀 분석
- 지금까지 내가 만든 클래스로 그래프를 띄워보자

일변량 모델은 한번에 하나의 특징만 process할 수 있다. 한가지의 feature을 선택하면 나머지는 무시 해야한다.

- Age, Thrust, Terameters를 각각 feature로 선택하여 3가지 그래프를 그려보자

- csv파일 읽어오는건 pandas를 이용하겠다.
- <img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg21.png" alt="MLimg0"  />

matplot의 plot에 사용할 수 있는 색이다.

plt.grid()로 격자를 그릴 수 있다.

```python
csv_data = pd.read_csv("../resources/spacecraft_data.csv")
print(csv_data["Age"])
x = np.array(csv_data["Age"]).reshape(-1,1)
y = np.array(csv_data["Sell_price"]).reshape(-1,1)
plt.scatter(x, y,color="navy", s=10)
plt.xlabel('x1: age (in years)')
plt.ylabel('y : Sell_price(in kiloeuros)')
plt.title('Plot of the selling prices of spacecrafts with respect to their age')
mylr = MyLinearRegression([[1.], [1.]])
mylr.fit_(x, y)
print(mylr.thetas)
print(mylr.cost_(x, y))
plt.scatter(x, mylr.predict_(x), color="cornflowerblue", s=7)
plt.legend(['Sell_price', 'Predicted sell_price'])
plt.grid()
plt.show()
```

코드는 이렇게 구현하였다.

### Age 그래프 그리기

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg22.png" alt="MLimg0"  />

alpha=0.0001, n_cycle=1020000로 하니 한번 모델을 적용하는데 1분이 걸린다...

cycle을  좀 줄여야 겠다.

cost fucntion은 24000정도가 제일 낮은것 같다.

thetas(1,1)로 시작했을때이다.. 앞으로는 점을 찍고 thetas를 추정하여 초기 thetas값을 정해야 겠다.



### Thrust power 그래프 그리기

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg23.png" alt="MLimg0"  />

앞의 age보다 상관관계가 더 뚜렷하다.



### Total distance 그래프 그리기`

수렴속도를 빨리 하기 위해 thetas를 추정할 것이다. 그러기 위해 우선 실제 값을 찍어보자

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg24.png" alt="MLimg0"  />

얼핏 기울기가 음수일 것같다. theta를 추정해보자 theta0은 1000, theta1은 -1로 추정해보겠다.

theata를 추정했으니 n_cycle을 줄였다.

그래프 title을 실수로 수정을 안해서 모두 동일하지만 위에는 age 중간은 thrust,  마지막은 dist이다

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg25.png" alt="MLimg0"  />

꽤나 유사하다.

그래프를 그리니 뭔가 정말 머신러닝을 하는것 같아 재미있다.

저 직선이 예측을 하고있지만 너무 부정확하다 이를 보완 해야 할 것 같다.

이제 다변량 회귀분석으로 넘어가보자



## Part two(Multivariate Linear Regression)

### 목표

- Train a single multivariate linear regression model on all three features.
- 결과 theta parameters을 Display하고 설명하자
- price 예측에서 각각 feature가 어떤 영향을 주는가?
- Mean Squared Error(MSE)를 이용하여 평가한다. Part one과 결과를 비교해보자



```python
mylr = MyLinearRegression([[338.94317973],
 [-22.67763021],
 [  5.84252624],
 [ -2.59281776]])
mylr.fit_(x, y)
plt.scatter(x_dist, y,color="purple", s=10)
plt.xlabel('x1: Terameters (in Tmeters)')
plt.ylabel('y : Sell_price(in kiloeuros)')
plt.title('Plot of the selling prices of spacecrafts with respect to their dist')
plt.scatter(x_dist, mylr.predict_(x), color="violet", s=7)
plt.legend(['Sell_price', 'Predicted sell_price'])
plt.grid()
plt.show()
```

수없는 반복끝에 theta값을 구해서 그래프를 그렸다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg26.png" alt="MLimg0"  />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg27.png" alt="MLimg0"  />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg28.png" alt="MLimg0"  />

앞선 Univariate Linear Regression보다 굉장히 정확하다 !!



## ex09

Question Time !!

**1 -What is the main diﬀerence between univariate and multivariate linear regression, in terms of variables?**

- 예를 들어 날씨를 판단할때 온도variable 하나로 날씨를 예측하는것보다 온도, 습도, 기압등 여러가지 요인을 기준으로 예측하는게 더정확하다. univariate는 온도하나로 날씨를 예측하는 것이고 multivariate는 온도뿐아니라, 습도, 기압, 조도등 여러 요소의 영향력을 더 추가하는것이다.
  이로써 multivariate는 상대적으로 더 정확한 예측이 가능하지만, 변수의 개수 만큼 학습 시간이 늘어난다.

**2 - Is there a minimum number of variables needed to perform a multivariate linear regression? If yes, which one?**

- 변수는 최소 2개여야 multivariate linear regression이라 부를 수 있다.

**3 - Is there a maximum number of variables needed to perform a multivariate linear regression? If yes, which one?**

- 변수의 최대 수는 제한이 없다. 하지만 모든 변수를 학습시키기엔 시간이 많이 걸리므로 유의미한 상관관계를 가지는 변수를 선택할 필요가 있다.

**4 - Is there a diﬀerence between univariate and multivariate linear regression in terms of performance evaluation?**

- 적어도 위에서 실행한것에 따르면 엄청난 차이가 있다. multivariate는 훨씬 더 정확하다.

**5 - What does it mean geometrically to perform a multivariate gradient descent with two variables?**

- univariate 에선 cost함수의 최솟값을 찾기위해 cost함수를 미분하여 기울기가 0을 만족하는 thetas를 찾아가는 과정이었다.(포물선에서 극솟값)
  이를  2가지 변수로 확장시키면(3차원 포물선 에서의 극솟값일 것이다.)

  <img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg29.png" alt="MLimg0" style="zoom:50%;" />

기하학적으로 해석한 예이다





##  Introducing Polynomial Models

지금까지 우리는 선형으로 모델을 만들었다.

하지만 아래와 같이 예측변수가 선형관계가 아닐 땐 무슨 방법을 써야할까?

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg30.png" alt="MLimg0"  />

이는 적절한 직선으로 예측할 수없다. 여기서 다항식(polynomial)을 이용하여 모델을 만들어보자 !!

x뿐아니라 x^2, x^3, x^n 까지 이용해 볼수 있다.



#### **Polynomial Hypothesis**

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg31.png" alt="MLimg0"  />

다항식으로 Hypothesis를 만들었다.



## ex10

##  Polynomial models

역시 numpy만 이용해야한다

지금까지 전부 sklearn 라이브러리가 금지되어 있는데 이걸 사용하면 내가 지금까지 구현한 걸 보다 쉽게 할 수 있는 것 같다.

### 목표

- Create a function that takes a vector x of dimension m∗1 and an integer n as input, and returns a matrix of dimension m∗n.

```python
def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args: x: has to be an numpy.ndarray, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised., →
    Returns: The matrix of polynomial features as a numpy.ndarray,
        of dimension m * n, containg he polynomial feature values for all training examples.
    None if x is an empty numpy.ndarray.
    Raises: This function should not raise any Exception. """
    temp = x.copy()
    for i in range(2, power + 1):
        temp = np.append(temp, np.power(x, i), axis=1 )
    return temp
```

numpy.append와 numpy.power을 이용하여 구현하였다.



## ex11

Polynomial Models을 Train 하자

### 목표

다항식 모델을 훈련시켜 보자

day01에서 사용했던 are_blue_pills_magic.csv파일을 이용한다.

변수를 하나를 이용해서 2 ~ 10차 다항식 Hypothesis를 세워서 cost를 평가해보자

즉  univariate Plynomial regression를 적용해보자

각각의 모델의 cost를 bar형태로 비교해서 best hypothsis를 정해보자

```
[[88.5],[-9 ]]
```

이전에 구했던 theta0 , theta1 은 이 두값으로 시작 하고 나머지 값들은 0으로 시작하겠다.

10차다항식 까지 있기 때문에 n_cycles는 10000으로 통일하겠다.

다항식의 exponet가 바뀔때마다 alpha값을 줄여주지 않으면 Nan으로 발산해 버린다.

이유는 9제곱까지 가면 데이터 크기가 조금만 커져도 발산해 버리기 때문이다.

따라서 데이터를 정규화 시켜주겠다.

이전에 배웠던 minmax와 z_score 두가지로 정규화를 진행하고 결과를 보자



--------------------------

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg33.png" alt="MLimg0"  />

처음에 alpha와 cycles를 잘못 조절하여 n==4가 이상적으로 나왔으나 이어지는 문제에서 원하는 결과가 아니기에 다시 n ==9일때를 이상적으로 맞추기 위해 노력했다..**

```
[[  0.99770029]
 [ -8.45545659]
 [ 38.01261233]
 [-46.05203096]
 [-16.20216812]
 [ 24.6999249 ]
 [ 26.22825386]
 [ -0.08605958]
 [-19.69747269]
 [  0.53536266]]
```

이 theta값을 얻기 위해서 정말 오래걸렸다 ㅠㅠ

---------------------------



<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg32.png" alt="MLimg0"  />

**minmax**로 정규화 했을시 그래프이다. 내 생각이 맞다면 n이 커질수록 정확할 것이다.

그러나 alpha값과 cycle을 일일히 테스트하는데 시간이 너무 오래걸린다.

n == 9일때 cost를 최소로 하는 theas를찾아보자

기존 데이터의 음의값이 없다면 minmax로 정규화 하는것이 좋아 보인다.

n=9일때 가장 cost가 가장 낮다. 구현하면서 느낀점은 각각의(x^n)의 계수의 alpha를 각각 try and error 기법으로 따로 따로 정한다면 더 정확하게 모델을 예측할 수 있을것 같다.(시간이 많이 걸리겠지만)

아래는 코드이다

```python
csv_data = pd.read_csv("../resources/are_blue_pills_magics.csv")
y_n = []
x = np.array(csv_data["Micrograms"]).reshape(-1,1)
x = zscore(x)

y = np.array(csv_data["Score"]).reshape(-1,1)
y = zscore(y)

x2 = add_polynomial_features(x, 2)
x3 = add_polynomial_features(x, 3)
x4 = add_polynomial_features(x, 4)
x5 = add_polynomial_features(x, 5)
x6 = add_polynomial_features(x, 6)
x7 = add_polynomial_features(x, 7)
x8 = add_polynomial_features(x, 8)
x9 = add_polynomial_features(x, 9)

mylr2 = MyLinearRegression([[88.85],[-9.0 ], [0]])
mylr3 = MyLinearRegression([[88.85],[-9.0 ], [0], [0]])
mylr4 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0]])

mylr5 = MyLinearRegression([[88.85],[-9.0 ], [0], [0], [0.0], [0.0]])
mylr6 = MyLinearRegression([[88.85],[-9.0 ], [0], [0.0], [0.0], [0.0], [0.0]])

mylr7 = MyLinearRegression([[88.85],[-9.0 ], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
mylr8 = MyLinearRegression([[88.85],[-9.0 ], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

mylr9 = MyLinearRegression([[88.85],[-9.0 ], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
mylr2.fit_(x2, y)
y_n.append(mylr2.cost_(x2,y))

mylr3.fit_(x3, y)
y_n.append(mylr3.cost_(x3,y))

mylr4.fit_(x4, y)
y_n.append(mylr4.cost_(x4,y))

mylr5.fit_(x5, y)
y_n.append(mylr5.cost_(x5,y))

mylr6.fit_(x6, y)
y_n.append(mylr6.cost_(x6,y))

mylr7.fit_(x7, y)
y_n.append(mylr7.cost_(x7,y))

mylr8.fit_(x8, y)
y_n.append(mylr8.cost_(x8,y))

mylr9.fit_(x9, y)
y_n.append(mylr9.cost_(x9,y))

plt.bar(range(2, 10), y_n)

plt.xlabel('n exponent')
plt.ylabel('cost')
plt.title('train nine separate Linear Regression models with polynomial hypotheses with degrees ranging from 2 to 10.')
plt.show()

```





## Plotting Curves With Matplotlib



We asked you to plot straight lines in the day00. Now you are working with polynomial models, the hypothesis functions are not straight lines, but curves. Plotting curves is a bit more tricky, because if you do not have enough data point, you will get an ugly broken line instead of a smooth curve. Here’s a way to do it.

```python
x = np.arange(1,11).reshape(-1,1)
y = np.array([[ 1.39270298], [ 3.88237651], [ 4.37726357], [ 4.63389049], [ 7.79814439], [ 6.41717461], [ 8.63429886], [ 8.19939795], [10.37567392], [10.68238222]])
plt.scatter(x,y)
plt.show()

from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR
# Build the model:
x_ = add_polynomial_features(x, 3)
my_lr = MyLR(np.ones(4).reshape(-1,1))
my_lr.fit_(x_, y)
## To get a smooth curve, we need a lot of data points
continuous_x = np.arange(1,10.01, 0.01).reshape(-1,1)
x_ = add_polynomial_features(continuous_x, 3)
y_hat = my_lr.predict_(x_)
plt.scatter(x,y)
# print(my_lr.thetas)
plt.plot(continuous_x, y_hat, color='orange')
plt.show()
```

예제가 또 잘못적혀져 있어서 조금 헤맸다.



<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg34.png" alt="MLimg0"  />

곡선그리기를 완료하였다.



## ex12

### Let’s PLOT some Polynomial Models!



- For each model you built in the last exercise, plot its hypothesis function h(θ) on top of a scatter plot of the original data points (x,y).



	ex11의 9개의 모델인것 같다.. 여기서 가장 이상적이였던 n == 9일때의 모델을 선택해서 위에  h(θ)를 그려 보겠다.

**데이터 정규화 후 플랏!**

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg35.png" alt="MLimg0" style="zoom:50%;" />

n == 4일 때

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg48.png" alt="MLimg0" style="zoom:50%;" />

(이 그래프는 overfit을 위해 아래에서 엄청난 train 끝에 얻은 그래프이다 ㅠㅠ)

n== 9일 때

여기서 y값을 정규화 하면 alpha값을 너무 줄이지 않아도 되는점을 깨달았다.

```python
고찰
우선 데이터를 정규화 했을 때,
x값이 0.1이라 하자
x^2, x^3 .. x^9
0.01 0.001 0.000000001
이 된다. 상대적으로 x^2보다 x^9이 aplha(learning rate)에 민감하게 반응할텐데 왜 pdf에서는 alpha를 동일하게 사용하는지 모르겠다 각각마다 learning rate를 다르게 주면 더 정확하게 측정할 수 있을 것 같다.
alpha가 너무 크면 overshoot가 일어나 발산이되고, alpha가 너무 작으면 수렴까지의 속도가 너무 오래걸린다.

따라서 작은 learning rate부터 알고리즘 연산을 해서 점점 큰 learning rate를 선택해야한다.
 ex)
0.001 -> 0.003 -> 0.01 -> 0.03 -> 0.1 -> 0.3 -> 1

이 문제에서 출제자의 의도는 다항식의 degree 가 높아질 수록 cost는 확실히 낮아지지만 이는 우리가 가지고 있는 데이터에 fit된것 뿐이고 전체 데이터와는 오차가 큰
overfit이 일어나는 상황을 의도해서
다항식의 차수가 높다고 해서 좋은 모델이 아니라는것을 알려주려고 하는 것 같다.
하지만 overfit인 상황을 만들기 위해선 많은 시간이 필요하다 ...

train을 빠르게 하기위해 반복문을 최적화 시켰다.
  while n_cycle:
            self.thetas -= self.alpha * ((self.gradient(x, y)))
            # for i, v in enumerate(self.gradient(x, y)):
            #     self.thetas[i] -= (self.alpha * v)
            n_cycle -= 1
        return self.thetas
```

overfitting을 일으키기 위해 시간을 제법 많이썻다..

아래는 학습 과정이다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg39.png" alt="MLimg0" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg40.png" alt="MLimg0" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg41.png" alt="MLimg0" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg42.png" alt="MLimg0" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg43.png" alt="MLimg0" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg44.png" alt="MLimg0" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg45.png" alt="MLimg0" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg46.png" alt="MLimg0" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg47.png" alt="MLimg0" style="zoom:50%;" />







<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg48.png" alt="MLimg0" style="zoom:50%;" />

**overfit을 일으키기 위해선 정말 시간을 많이 써야한다 ㅠㅠ**

-  Answer the following questions:
   - 1. From a purely intuitive point of view, which hypothesis (i.e. which polynomial degree) seems to best represent the relationship between y and x?
        -  직관적으로는 n이 4일때 n이 9일때보다 오히려 유사해 보인다..
   - 2. Go back to your answer for the previous exercise. What polynomial degree had you identiﬁed as the most accurate on the dataset? Is it the same one as the curve you just picked as the most representative of the relationship between y and x?
        - 아니요 .. cost는 n==9일때가 가장 낮지만 직관적을 보기엔 n==4가 유사해 보입니다.
   - 3. Find the plot that illustrates that same hypothesis you had found was the most accurate in the previous exercise. Take a closer look at the curve. What do you think? Does it seem to properly represent the relationship between y and x?
        - 각각의 데이터와는 예측값이 유사하지만 전체적인 흐름은 이상하다.



## Lost in Overﬁtting

```
The two previous exercises lead you, dear reader, to a very dangerous territory: the realm of overﬁtting. You did not see it coming but now, you are in a bad situation... By increasing the polynomial degree of your model, you increased its complexity. Is it wrong? Not always. Some models are indeed very complex because the relationships they represent are very complex as well. But, if you look at the plots for the previous exercise’s best model, you should feel that something is wrong
```

맞다 overfitting이란게 일어나서 굉장히 모양이 이상해졌다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg48.png" alt="MLimg0" style="zoom:50%;" />

학습데이터에서만 오차가 감소하고 실제 데이터에선 오차가 증가하게된다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg37.png" alt="MLimg0" style="zoom:50%;" />

이러한 모델은 새로운 데이터에 대해선 오차가 큰 예측을 할 수 있다.

**그렇기 때문에 데이터셋은 교육용과 학습용 두가지가 필요하다 **





## ex13

###  DataSpliter

### 목표

Learn how to split a dataset into a **training set** and a **test set.**



데이터셋을 training set과 test set으로 shuffles 과 splits하는 함수를 구현하자

- train용 데이터를 일정 비율 유지하고 나머지를 test용 데이터로 사용한다.
- y vector(실제 출력값)도 나눠진 x에 맞게 shuffles, splits 해야 한다.



주의

- 데이터를 분리하기전에 shuffled 작업을 진행한다.

- Unless you use the same seed in your randomization algorithm, you won’t get the same results twice.



**두가지 배열의 순서를 동시에 지키면서 랜덤으로 섞을 수 있는 방법은 무엇일까?**

numpy에선 이런식의 참조가 가능하다.

```python
x = np.arrange(10)
x[[8,9]]
>> [8,9]
```

즉 동시에 여러 인덱스를 입력하여 참조가 가능하다. 이를이용하면

```python
x = np.arange(10)
y = np.arange(10)

s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]
```

이런식으로 배열을 직접 섞는게 아닌 index배열을 만들어서 shuffle한 후 각각 참조하면 된다.
이렇게하면 x , y 의 순서가 완벽히 유지되면서 shuffle을 할 수 있다.

최종 코드이다

```python
def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the traning set., →
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1. proportion: has to be a float,
    the proportion of the dataset that will be assigned to the training set., →
    Returns: (x_train, x_test, y_train, y_test) as a tuple of numpy.ndarray None if x or y is an empty numpy.ndarray.
    None if x and y do not share compatible dimensions.
    Raises: This function should not raise any Exception. """
    n = int(float(x.shape[0] * proportion))
    idxs = np.arange(x.shape[0])
    np.random.shuffle(idxs)
    x_train = x[0:n]
    x_test = x[n:]
    y_train = y[0:n]
    y_test = y[n:]
    return (x_train, x_test, y_train, y_test)
```



## ex14

**Let’s do Machine Learning for real!**



csv파일을 training set과 test set으로 분리한다.

이전에 만들었던 polynomial_features method를 training set에 적용한다.

Linear Regression models with polynomial hypotheses를 degrees ranging from 2 to 10을 적용해서 모델을 평가한다.

이제 다시 최고의 hypothesis를 골라보자

최댓값, 최솟값을 알땐 minmax를 이용하고

모를때는 모집단이 정규분포를 따른다고 가정하고 z_score을 이용하다

이번엔 데이터를 나누었으므로 minmax가아닌 z_score를 이용하여  정규화 한 후 학습시키겠다.



**아래는 학습시키기 위한 코드이다.**

```python
csv_data = pd.read_csv("../resources/are_blue_pills_magics.csv")
y_n = []
x = np.array(csv_data["Micrograms"]).reshape(-1,1)
x = zscore(x)

y = np.array(csv_data["Score"]).reshape(-1,1)
y = zscore(y)

temp = data_spliter(x, y, 0.5)
x_train = temp[0]
x_test = temp[1]
y_train = temp[2]
y_test = temp[3]

x2 = add_polynomial_features(x_train, 2)
x3 = add_polynomial_features(x_train, 3)
x4 = add_polynomial_features(x_train, 4)
x5 = add_polynomial_features(x_train, 5)
x6 = add_polynomial_features(x_train, 6)
x7 = add_polynomial_features(x_train, 7)
x8 = add_polynomial_features(x_train, 8)
x9 = add_polynomial_features(x_train, 9)

x2_test = add_polynomial_features(x_test, 2)
x3_test = add_polynomial_features(x_test, 3)
x4_test = add_polynomial_features(x_test, 4)
x5_test = add_polynomial_features(x_test, 5)
x6_test = add_polynomial_features(x_test, 6)
x7_test = add_polynomial_features(x_test, 7)
x8_test = add_polynomial_features(x_test, 8)
x9_test = add_polynomial_features(x_test, 9)

mylr2 = MyLinearRegression([[88.85],[-9.0 ], [1]])
mylr3 = MyLinearRegression([[88.85],[-9.0 ], [1], [1]])
mylr4 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1]])

mylr5 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1]])
mylr6 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1]])

mylr7 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1], [1]])
mylr8 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1], [1], [1]])

mylr9 = MyLinearRegression([[88.85],[-9.0 ], [1], [1], [1], [1], [1], [1], [1], [1]])

mylr2.fit_(x2, y_train)
mylr2.alpha = 0.00001
mylr2.fit_(x2, y_train)
mylr2.alpha = 0.00003
mylr2.fit_(x2, y_train)
mylr2.alpha = 0.0001
mylr2.fit_(x2, y_train)
mylr2.alpha = 0.0003
mylr2.fit_(x2, y_train)
y_n.append(mylr2.cost_(x2_test,y_test))

mylr3.fit_(x3, y_train)
mylr3.alpha = 0.00001
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.00003
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.0001
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.0003
mylr3.fit_(x3, y_train)
mylr3.alpha = 0.001
mylr3.fit_(x3, y_train)
y_n.append(mylr3.cost_(x3_test,y_test))

mylr4.fit_(x4, y_train)
mylr4.alpha = 0.00001
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.00003
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.0001
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.0003
mylr4.fit_(x4, y_train)
mylr4.alpha = 0.001
mylr4.fit_(x4, y_train)
y_n.append(mylr4.cost_(x4_test,y_test))

mylr5.fit_(x5, y_train)
mylr5.alpha = 0.00001
mylr5.fit_(x5, y_train)
mylr5.alpha = 0.00003
mylr5.fit_(x5, y_train)
mylr5.alpha = 0.0001
mylr5.fit_(x5, y_train)
y_n.append(mylr5.cost_(x5_test,y_test))

mylr6.fit_(x6, y_train)
mylr6.alpha = 0.00001
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.00003
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.0001
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.0003
mylr6.fit_(x6, y_train)
mylr6.alpha = 0.001
mylr6.fit_(x6, y_train)
y_n.append(mylr6.cost_(x6_test,y_test))

mylr7.fit_(x7, y_train)
mylr7.alpha = 0.00001
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.00003
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.0001
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.0003
mylr7.fit_(x7, y_train)
mylr7.alpha = 0.001
mylr7.fit_(x7, y_train)
y_n.append(mylr7.cost_(x7_test,y_test))

mylr8.fit_(x8, y_train)
mylr8.alpha = 0.000003
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.00001
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.00003
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.0001
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.0003
mylr8.fit_(x8, y_train)
mylr8.alpha = 0.001
mylr8.fit_(x8, y_train)
y_n.append(mylr8.cost_(x8_test,y_test))

mylr9.alpha = 0.0000001
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.0000003
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.0000008
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.000001
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.000003
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.000008
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.00001
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.00003
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.00008
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.0001
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.0003
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.0008
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.001
mylr9.fit_(x9, y_train)
mylr9.alpha = 0.003
mylr9.fit_(x9, y_train)
y_n.append(mylr9.cost_(x9_test,y_test))
print(mylr9.thetas)
print(mylr9.cost_(x9_test,y_test))
plt.bar(range(2, 10), y_n)

plt.xlabel('n exponent')
plt.ylabel('cost')
plt.title('train nine separate Linear Regression models with polynomial hypotheses with degrees ranging from 2 to 10.')
plt.show()


```

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg49.png" alt="MLimg0" style="zoom:50%;" />





사실 각각의 모델마다 최소의 cost가 나올때까지 train을 시킬 수있지만 이는 너무 오래 걸리기 때문에 어느 정도까지만 train을 시켜보았다.  9차다항식은 발산해버렸다.. (n_cycles를 너무 적게주었다. )

train과 testset을 따로 두었을땐 **3차 polynomial hypotheses** 가설이 가장 알맞다.

degree가 높다고 무조건 좋은 모델은 아닌것이다.



## ex15

### Question Time!

**1 - What is overﬁtting?**

- 모집단이 아닌 train데이터에만 cost함수가 최적으로 맞춰진것이다. 이 경우 실제 새로운 데이터값이 들어오면 오차가 커질 수 있다.

**2 - What do you think underﬁtting might be?**

- **overfitting**이 다항식이 복잡해져서 일어났다면 **underfitting**은 다항식이 너무 단순하여 오차가 줄어들지 않는 경우를 말할 것 같습니다.



**3 - Why is it important to split the data set in a training and a test set?**

- 새로운 데이터를 테스트 해 봐서 **overfitting**을 막기 위함입니다.



**4 - If a model overﬁts, what will happen when you compare its performance on the training set and the test set?**

- training set에서는 최고의 효율로 예측하겠지만 test set에서는 오차가 많이 발생합니다.

**5 - If a model underﬁts, what do you think will happen when you compare its performance on the training set and the test set?**

- underfitting이 일어난다면 test set과 training set 모두 오차가 많이 발생할 것 같습니다.

[깃허브 링크](https://github.com/ChoiDongKyu96/Bootcamp_Machine_Learning)