---
layout: post
category: ai
tags:
- AI
- MachineLearning
excerpt: 머신러닝 배우기 네번째 걸음
comments: true
title: Bootcamp Machine Learning day03
---

# Bootcamp Machine Learning day03 

- 머신러닝 기초공부 day03에서 느낀점을 적는다.



  * [Bootcamp Machine Learning day03](#bootcamp-machine-learning-day03)
    * [ex00](#ex00)
      * [Multivariate Linear Regression with Class](#multivariate-linear-regression-with-class)
    * [ex01](#ex01)
      * [DataSpliter](#dataspliter)
    * [ex02](#ex02)
    * [Interlude - Classiﬁcation: The Art of Labelling Things](#interlude---classiﬁcation-the-art-of-labelling-things)
      * [다시한번 !!](#다시한번-)
    * [Interlude - Predict I: Introducing the Sigmoid Function](#interlude---predict-i-introducing-the-sigmoid-function)
      * [Prediction start](#prediction-start)
        * [Formulating a Hypothesis](#formulating-a-hypothesis)
      * [ex03](#ex03)
    * [Interlude - Predict II : Hypothesis](#interlude---predict-ii--hypothesis)
      * [Logistic Hypothesis](#logistic-hypothesis)
    * [ex04](#ex04)
      * [<strong>Logistic Hypothesis</strong>](#logistic-hypothesis-1)
    * [Evalutate](#evalutate)
      * [<strong>case 1 : the expected output is 1</strong>](#case-1--the-expected-output-is-1)
      * [<strong>Case 2: The expected output is 0</strong>](#case-2-the-expected-output-is-0)
      * [Cross-entropy](#cross-entropy)
    * [ex05](#ex05)
    * [Logistic Loss Function](#logistic-loss-function)
    * [Interlude - Linear Algebra Strikes Again!](#interlude---linear-algebra-strikes-again)
    * [ex06](#ex06)
      * [Vectorized Logistic Loss Function](#vectorized-logistic-loss-function)
    * [Improve](#improve)
      * [The logistic gradient](#the-logistic-gradient)
    * [ex07](#ex07)
      * [Logistic Gradient](#logistic-gradient)
    * [ex08](#ex08)
    * [ex09](#ex09)
      * [Logistic Regression](#logistic-regression)
    * [ex10](#ex10)
      * [Practicing Logistic Regression](#practicing-logistic-regression)
      * [Part 1 - One Label to Discriminate Them All](#part-1---one-label-to-discriminate-them-all)
      * [Part 2 - One Versus All](#part-2---one-versus-all)
      * [구현 과정](#구현-과정)
   * [모델 생성 완료](#모델-생성-완료)
   * [전체 데이터 예측](#전체-데이터-예측)
   * [데이터 확률 최댓값을 기준으로 클래스 분류](#데이터-확률-최댓값을-기준으로-클래스-분류)
   * [시각화](#시각화)
     * [ex12](#ex12)
       * [Confusion Matrix](#confusion-matrix)



## ex00

### Multivariate Linear Regression with Class

day 02에서 구현했던 Multivariate Linear Regression with Class 이다.



```python
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    mylinearregression.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/27 16:39:10 by dochoi            #+#    #+#              #
#    Updated: 2020/05/28 16:48:58 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

class MyLinearRegression(object):
    """ Description: My personnal linear regression class to fit like a boss. """
    def __init__(self, thetas, alpha=0.000001, n_cycle=30000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

    def add_intercept(self,x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    def gradient(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        return self.add_intercept(x).T @ (self.predict_(x).reshape(-1,1) - y) / len(x)

    def fit_(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        n_cycle= self.n_cycle
        while n_cycle:
            self.thetas -= self.alpha * ((self.gradient(x, y)))
            # for i, v in enumerate(self.gradient(x, y)):
            #     self.thetas[i] -= (self.alpha * v)
            n_cycle -= 1
        return self.thetas

    def predict_(self,x):
        if x.ndim == 1:
            x = x[:,np.newaxis]
        if len(self.thetas) - 1 != x.shape[1]  or len(x) == 0:
            return None
        return self.add_intercept(x) @ self.thetas

    def cost_elem_(self, x, y):
        y_hat = self.predict_(x)
        if y.ndim == 1:
            y = y[:,np.newaxis]
        if y.shape != y_hat.shape or len(y) == 0 or len(y_hat) ==0:
            return None
        return ((y - y_hat) ** 2) / (2 * len(y))

    def cost_(self, x, y):
        y_hat = self.predict_(x)
        if len(y) == 0 or len(y_hat) == 0 or y.shape != y_hat.shape:
            return None
        return (sum((y_hat - y) * (y_hat - y)).squeeze() / (2 * len(y)))
```



## ex01

### DataSpliter

day 02에서 구현한 test set 과 train set으로 나눠주는 함수이다.

```python
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_spliter.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/28 15:08:21 by dochoi            #+#    #+#              #
#    Updated: 2020/05/28 16:33:20 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

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



## ex02

day02에 있던 질문들이다.



***\*1 -What is the main diﬀerence between univariate and multivariate linear regression, in terms of variables?\****



\- 예를 들어 날씨를 판단할때 온도variable 하나로 날씨를 예측하는것보다 온도, 습도, 기압등 여러가지 요인을 기준으로 예측하는게 더정확하다. univariate는 온도하나로 날씨를 예측하는 것이고 multivariate는 온도뿐아니라, 습도, 기압, 조도등 여러 요소의 영향력을 더 추가하는것이다.

 이로써 multivariate는 상대적으로 더 정확한 예측이 가능하지만, 변수의 개수 만큼 학습 시간이 늘어난다.



***\*2 - Is there a minimum number of variables needed to perform a multivariate linear regression? If yes, which one?\****



\- 변수는 최소 2개여야 multivariate linear regression이라 부를 수 있다.



***\*3 - Is there a maximum number of variables needed to perform a multivariate linear regression? If yes, which one?\****



\- 변수의 최대 수는 제한이 없다. 하지만 모든 변수를 학습시키기엔 시간이 많이 걸리므로 유의미한 상관관계를 가지는 변수를 선택할 필요가 있다.



***\*4 - Is there a diﬀerence between univariate and multivariate linear regression in terms of performance evaluation?\****



\- 적어도 위에서 실행한것에 따르면 엄청난 차이가 있다. multivariate는 훨씬 더 정확하다.



***\*5 - What does it mean geometrically to perform a multivariate gradient descent with two variables?\****



\- univariate 에선 cost함수의 최솟값을 찾기위해 cost함수를 미분하여 기울기가 0을 만족하는 thetas를 찾아가는 과정이었다.(포물선에서 극솟값)

 이를 2가지 변수로 확장시키면(3차원 포물선 에서의 극솟값일 것이다.)





## Interlude - Classiﬁcation: The Art of Labelling Things

지금까지 머신러닝 알고리즘을 구현했다. Predict > Evaluate > Improve의 무한순환..

 **Multivariate Linear Regression**

- can now be used to predict a numerical value, based on several features. This algorithm uses gradient descent to optimize its cost function



**classiﬁcation algorithm(Logistic Regression)**

- Logistic Regression
  - 이걸로는 숫자값(나이, 성적, 가격 등)이 아니라 categories, or labes(like 개, 강아지, 질환)같은것을 예측할 수 있다.



주의사항

Logistic Regression에서의 '**regression**'은 회귀가 아니라 분류작업이다. 



### 다시한번 !!

 **Logistic Regression** 은 **classiﬁcation algorithm**(분류알고리즘) 이다. !!

( Multivariate Linear Regression와  다르다)



In this bootcamp we will use the following terms interchangeably: class, category, and label. They all refer to the groups to which each training example can be assigned to, in a classiﬁcation task.



클래스, 카테고리, 라벨을 같은언어로 사용한다고 한다.

![](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg50.png)

## Interlude - Predict I: Introducing the Sigmoid Function

### Prediction start

#### Formulating a Hypothesis

hypothesis, 즉  h(θ)는 feature set으로 이루어진 방정식이다.

prediction을 output할 때 사용한다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg31.png" alt="MLimg0"  />

**Polynomial Hypothesis**이다.

이걸로 환자가 아픈지 안아픈지 예측할 수 있을까?
아픈 환자에게는 1의 값을 주고 건강한 환자에게 0의 값을 주어
환자가 아플 확률을 0~1의 float값으로 출력할 수 있을 것이다.
앞에서 구현했던 선형방식을 그대로 사용하면 된다.

 All we need to do is sqash its output through another function that is bounded between 0 and 1. That’s the Sigmoid function and your next exercise is to implement it!



This function is also known as Standard logistic sigmoid function. This explains then a name logistic regression. The sigmoid function transforms an input into a probability value, i.e. a value between 0 and 1. This probability value will then be used to classify the inputs.

### ex03

**Sigmoid**를 구현하자

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg51.png" alt="MLimg0"  />

이함 수는 입력을 확률값, 즉 0과 1 사이의 값으로 변환한다. 이 확률 값은 입력을 분류하는데 사용된다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg56.png" alt="MLimg0"  />

이 sigmoid 함수는 logistic function의 특수한 경우이다. 

e는 자연상수 2.718....이다.

```python
import math
import numpy as np


def sigmoid_(x):
    """ Compute the sigmoid of a vector.
    Args:
    x: has to be an numpy.ndarray, a vector
    Returns: The sigmoid value as a numpy.ndarray.
    None if x is an empty numpy.ndarray.
    Raises: This function should not raise any Exception. """
    if len(x) == 0:
        return None
    def func(x):
        return 1 / (1 + pow(math.e, -x))
    return np.vectorize(func)(x)
```

math의 자연상수 e를 이용하고 vectorize를 통해 x안의 원소에 대해 함수를 적용해 주었다.

사실 numpy의 broadcasting때문에 vectorize를 적용하지 않아도 동일한 결과 나온다.
ex04에서 더욱 간단하게 만들어본다





## Interlude - Predict II : Hypothesis

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg53.png" alt="MLimg0"  />



sigmoid함수를 plot한것이다. output values가 0에서 1사이로 나타난다. 어떤 수를 넣더라도 output은 이 범위 안에서 출력된다.

### Logistic Hypothesis

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg54.png" alt="MLimg0"  />

**Logistic regression hypothesis**이다.

**linear regression hypothesis**에 비하면 정말 간단하다 

이걸 vectorize하면

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg55.png" alt="MLimg0"  />

이렇게 표현할 수 있다.

다시 살펴보자

**sigmoid function**은 [0,1]로 mapping해주는 함수이다.

이는 individual을 주어진 class의 member가 될 확률로 해석할 수 있게 해준다.



## ex04

###  **Logistic Hypothesis**

vectorize버전으로 구현한다.

```python
def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
Args:
    x:
        has to be an numpy.ndarray, a vector of dimension m * n.
    theta:
        has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    def add_intercept(x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    if (len(x) == 0 or len(theta) == 0 or x.shape[1] != (len(theta) - 1)):
        return None
    return 1 / (1 + pow(math.e, -add_intercept(x) @ theta))
```

**linear regression hypothesis**때 구현한 add_intercept, 즉 컬럼에 1을 추가하여 theta0원소를 한번에 계산할 수 있게 해주는 함수이다.



## Evalutate

Our model can predict the probability for a given example to be part of the class labeled as 1. Now it’s time to evaluate how good it is.

우리의 모델은 주어진 예제가 1로 labeled된 class의 part가 될 확률을 예측할 수 있다.

선형회귀에서 사용한(MSE) cost 함수는 적절하지 않다.

오로지 두가지로 분류한다고 가정할 때

- zero, if the element is not a member of the predicted class
- one, if the element is a member of the predicted class,



prediction과 label 사이의 'distance'를 측정하는 것은 classification model에선 좋은 평가 방법이 아니다. 우리는 logarithmic function(로그 함수)를 사용할 것이다. 왜냐하면 이는 잘못된 예측을 훨씬 더 알기 쉽기 때문이다.

### **case 1 : the expected output is 1**

- <img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg57.png" alt="MLimg0"  />

  y_hat이 0에 가까워지면 high cost가 나와야 한다.

  저 로그함수를 cost함수로 쓰면 어떨까?

  <img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg58.png" alt="MLimg0"  />

  prediction이 0에 가까워지면 cost가 급속도로 커지고 prediction이 1에 가까워지면 cost는 0에 가까워진다. 이를 통해 predction이 0에 가까워졌을때 harshly penalize를 할 수 있다.

  하지만 우리가 y_hat이 0에 가까워지길 원한다면 어떻게 해야할까?

  

### **Case 2: The expected output is 0**





<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg59.png" alt="MLimg0"  />

y_hat이 0이 되게 하고싶다.  아까 로그함수를 살짝 수정하면 가능하다.

아까와 반대의 모형이다.

prediction이 1에 가까워지면 cost가 급속도로 커지고 prediction이 0에 가까워지면 cost는 0에 가까워진다. 이를 통해 predction이 1에 가까워졌을때 harshly penalize를 할 수 있다.



이제 이 두모델을 어떻게 이용할지를 생각하자.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg60.png" alt="MLimg0"  />

두가지 그래프를 동시에 그렸다.

이를 if else을 이용해서 각각 y_hat이 0을 추구하길, 1을 추구하길 나눌 수 있지만

이쁘지 않고 하나의 수학 방정식으로 표현할 수 없다.

이를 하나의 방정식으로 통합하기 위해 수학적인 트릭을 이용한다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg61.png" alt="MLimg0"  />

이를 통합하자.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg62.png" alt="MLimg0"  />

**얼핏 보면 굉장히 복잡해 보이지만 cost에 수식을 대입한것 뿐이다.**

### Cross-entropy



모든 훈련 examples에 평균을 취하는 **cross-entrop y**라는 최종 cost fuction을 만들어야 한다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg63.png" alt="MLimg0"  />

바로 이것이다.

아래는 증명이다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg64.png" alt="MLimg0"  />1





## ex05

##  Logistic Loss Function

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg63.png" alt="MLimg0"  /> 

이 함수를 구현하자

주의사항은 log(0)은 정의할 수 없으므로 0을 대신할 epsilon(default=1e-15)를 정의한다.

이 함수를 **Cross-Entropy loss, or logistic loss** 라고 부른다

```python
def log_loss_(y, y_hat, eps=1e-15):
    """ Computes the logistic loss value. Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    eps: has to be a float, epsilon (default=1e-15)
    Returns: The logistic loss value as a float. None on any error.
    Raises: This function should not raise any Exception. """
    if (y.shape != y_hat.shape or len(y) == 0 or len(y_hat) == 0):
        return None
    def func(x):
        return math.log(x)
    def func2(x):
        return (math.log(1 - x))
    y_hat = y_hat + eps
    y_hat_log = np.vectorize(func)(y_hat)
    y_hat_log_inv = np.vectorize(func2)(y_hat)
    return -sum(y * y_hat_log + (1 - y) * y_hat_log_inv)  / len(y_hat)
```

numpy.log를 쓰면 간단하지만 수학적인 의미를 강조하기 위해 위와 같이 구현하였다.

ex06에서는 np.log로 더욱 간단하게 구현하겠다.



## Interlude - Linear Algebra Strikes Again!

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg52.png" alt="MLimg0"  />

이게 **Logistic Cost Function** 이다.  

벡터와 스칼라 사이의 덧셈, 뺄셈은 정의되지 않기 때문에 저 1벡터 표기법을 사용한다.

스칼라와 벡터사이에선 곱셈밖에 하지 못한다.



하지만 numpy에서는  관대하다.



```python
import numpy as np

y = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
ones = np.ones(y.shape[0	]).reshape((-1,1))
print(ones - y)

y = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
print(1 - y)
```

두가지의 output이 같다. 넘파이에서는 1을 자동으로 y의 차원과 같게 확장시켜준다.
이것이 numpy의 **Broadcasting**기능이다. 이는 매우 편리하지만 버그가 생길 수 있다.

```
Many of the bugs you will encounter while working on Machine Learning problems will come from NumPy’s permissiveness. Such bugs generaly don’t throw any errors, but mess they up the content of your vectors and matrices and you’ll spend an awful lot of time looking for why your model doesn’t learn. This is why we strongly suggest that you pay attention to your vector (and matrix) dimensions and stick as much as possible to the actual mathematical operations. 
```

머신러닝에서 마주치게 될 많은 버그는 numpy의 허용성에서 비롯된다.

이런 버그들은 일반적으로 어떤 error도 throw하지 않지만 벡터와 매트릭스 내용을 엉망으로 만들고 모델이 learn 되지 않는데 많은 시간을 쓰게한다. 따라서 벡터, 행렬 차원에 주의를 기울이고 가능한 실제 수학연산과 같게 코드를 작성해야 한다.



## ex06

### Vectorized Logistic Loss Function

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg65.png" alt="MLimg0"  />

이를 구현하자

```python
def vec_log_loss_(y, y_hat, eps=1e-15):
    """ Compute the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.
    eps: epsilon (default=1e-15)
    Returns: The logistic loss value as a float. None on any error.
    Raises: This function should not raise any Exception. """
    if (y.shape != y_hat.shape or len(y) == 0 or len(y_hat) == 0):
        return None
    y_hat = y_hat + eps
    return -sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)) ) / len(y_hat)
```

조금더 간단하게 구현하였다. 역시 eps은 log(0)이 정의되지 않기에, 버그가 생기는걸 방지하기 위함이다.



## Improve

이제 improve를 할 시간이다. 다른 의미론 prediction의 cost를 감소시키는 작업이다.

이번에도 미분하여 gradient를 계산한다.이는  θ parameters를 조정할 수 있게 할 것이다.

### The logistic gradient

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg66.png" alt="MLimg0"  />

logistic cost함수의 gradient는 linear regression gradient와 공식이 유사하다.

단지 cost함수의 형태가 다를 뿐이다. (logistic cost함수는 sigmoid이다.)

logistic cost함수의 gradient를 구하는건 조금 복잡해서 따로 깊은 설명을 해주지 않는다.

바로 결과만 제시해준다. 구현을 해보자

## ex07

### Logistic Gradient

```python
def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns: The gradient as a numpy.ndarray, a vector of dimensions n * 1,
    containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises: This function should not raise any Exception. """
    if len(x) == 0 or len(y) == 0 or len(theta) == 0 :
        return None

    return np.append(np.sum((logistic_predict_(x, theta) - y)) / len(x) , np.sum((logistic_predict_(x, theta) - y) * x, axis=0) / len(x)).reshape(-1 ,1)

```

행렬곱으로 구현하였다..

하지만 다시 문제를 읽어보니 forloop로 구현하라고 되어있다. 수학 연산과정을 확실히 이해하라는 뜻인것 같다.

다시 forloop를 이용하여 구현하였다.

```python
def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns: The gradient as a numpy.ndarray, a vector of dimensions n * 1,
    containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises: This function should not raise any Exception. """
    if len(x) == 0 or len(y) == 0 or len(theta) == 0 :
        return None
    y_hat = logistic_predict_(x, theta)
    answer = np.sum(y_hat- y)
    for j in range(0, len(theta) - 1):
        temp = 0.0
        for i in range(len(x)):
            temp += ((y_hat[i] - y[i]) * x[i][j])
        answer = np.append(answer, temp)
    return answer / len(x)
```



## ex08

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg67.png" alt="MLimg0"  />

transpose를 이용하여 더 간단한 식으로 구현이다.

```python
def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible dimensions.,
     Args:
     x: has to be an numpy.ndarray, a matrix of dimension m * n.
     y: has to be an numpy.ndarray, a vector of dimension m * 1.
     theta: has to be an numpy.ndarray, a vector (n +1) * 1.
     Returns: The gradient as a numpy.ndarray, a vector of dimension n * 1,
     containg the result of the formula for all j.,
     None if x, y, or theta are empty numpy.ndarray.
     None if x, y and theta do not have compatible dimensions.
     Raises: This function should not raise any Exception. """

    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    return add_intercept(x).T @ (logistic_predict_(x, theta) - y) / len(x)
```



## ex09

### Logistic Regression

logistic regression 을 class로 구현하자 물론 sklearn은 금지고 numpy만 허용된다.

```python
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_logistic_regression.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 02:00:53 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 02:54:49 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import math

class MyLogisticRegression(object):
    """ Description: My personnal logistic regression to classify things. """

    def __init__(self, thetas, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.n_cycle = n_cycle
        self.thetas = np.array(thetas, dtype=float).reshape(-1, 1)

    def add_intercept(self,x):
        if len(x) == 0 or x.ndim >= 3:
            return None
        if x.ndim == 1:
            return np.vstack((np.ones(len(x)), x)).T
        else:
            return np.insert(x, 0, 1, axis=1)

    def log_gradient(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        return self.add_intercept(x).T @ (self.predict_(x) - y) / len(x)

    def fit_(self, x, y):
        if len(x) == 0 or len(y) == 0 or len(self.thetas) == 0:
            return None
        n_cycle= self.n_cycle
        while n_cycle:
            self.thetas -= self.alpha * ((self.log_gradient(x, y)))
            n_cycle -= 1
        return self.thetas

    def predict_(self,x):
        if (len(x) == 0 or len(self.thetas) == 0 or x.shape[1] != (len(self.thetas) - 1)):
            return None
        return 1 / (1 + pow(math.e, -self.add_intercept(x) @ self.thetas))

    def cost_(self, x, y, eps=1e-15):
        y_hat = self.predict_(x) - eps
        if (y.shape != y_hat.shape or len(y) == 0 or len(y_hat) == 0):
            return None
        return -sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat)) ).squeeze() / len(y_hat)

```

구현을 완료하였다.



## ex10

###  Practicing Logistic Regression



이제 내가 만든 모델을 csv데이터를 불러와서 실제 적용해보자

Logistic Regression을 통해서 사람의 생체 정보로 어느 행성에서 왔는지 예측을 해보자

여기선 4가지로 분류해야한다.

하지만 지금까지 만든 모델은 0과 1 두가지 분류이다. 어떻게 해야할까?

### Part 1 - One Label to Discriminate Them All

행성 하나만 1로하고 나머지 전부를 0으로 해볼까?

1. 데이터셋을 테스트 와 훈련용으로 나눈다.
2. 행성하나만 1로 하고 나머지를 전부 0으로 하는 배열을 만든다.

3. 훈련한다. 

데이터를 **normalization** 해야한다. 왜냐하면 e^x꼴의 가설이기 때문에 숫자가 너무 커지거나 작아질 수 있다.

이제 **Multiclass Logistic Regression**는 어떻게 해야 하는가? 



### Part 2 - One Versus All

1. Part 1을 다른 행성에도 모두 적용해준다.
2. 모든 분류기에 class를 넣고 가장 확률이 높은것을 택한다.



**example :**

If a cititzen got the following classiﬁcation probabilities:

- Planet 0 vs all: 0.38 
- Planet 1 vs all: 0.51 
- Planet 2 vs all: 0.12 
- Planet 3 vs all: 0.89

Then the citizen should be classiﬁed as coming from Planet 3.



### 구현 과정

1. 데이터를 훈련셋과 평가셋으로 나눈다.

2. 데이터를 정규화한다. (y는 0,1,2,3 으로 이루어져 있기 때문에 정규화 하지 않는다.)

3. 4가지의 모델을 만든다.

4. Practicing Logistic Regression을 적용한다.

   

5. ```python
   # **************************************************************************** #
   #                                                                              #
   #                                                         :::      ::::::::    #
   #    log_reg_model.py                                   :+:      :+:    :+:    #
   #                                                     +:+ +:+         +:+      #
   #    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
   #                                                 +#+#+#+#+#+   +#+            #
   #    Created: 2020/05/30 02:56:12 by dochoi            #+#    #+#              #
   #    Updated: 2020/05/31 21:57:22 by dochoi           ###   ########.fr        #
   #                                                                              #
   # **************************************************************************** #
   
   import numpy as np
   import math
   import pandas as pd
   import matplotlib.pyplot as plt
   from my_logistic_regression import MyLogisticRegression as MyLR
   from data_spliter import data_spliter
   
   def zscore(x):
       if len(x) == 0:
           return none
       mu = sum(x)/len(x)
       temp = 0.0
       for elem in x:
           temp += ((elem - mu) * (elem - mu))
       var = temp
       std = np.sqrt(var/ (len(x) - 1))
       return (x - mu) / std
   
   csv_data_x = pd.read_csv("../resources/solar_system_census.csv")
   
   csv_data_y = pd.read_csv("../resources/solar_system_census_planets.csv")
   
   x = np.array(csv_data_x[["height","weight","bone_density"]])
   y =  np.array(csv_data_y["Origin"]).reshape(-1,1)
   
   x = zscore(x)
   
   
   temp = data_spliter(x, y, 0.5)
   x_train = temp[0]
   x_test = temp[1]
   y_train = temp[2]
   y_test = temp[3]
   
   y_train0 = np.array([1 if i == 0 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
   y_test0 = np.array([1 if i == 0 else 0 for i in y_test]).reshape(-1,1)
   
   y_train1 = np.array([1 if i == 1 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
   y_test1 = np.array([1 if i == 1 else 0 for i in y_test]).reshape(-1,1)
   
   y_train2 = np.array([1 if i == 2 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
   y_test2 = np.array([1 if i == 2 else 0 for i in y_test]).reshape(-1,1)
   
   y_train3 = np.array([1 if i == 3 else 0 for i in y_train]).reshape(-1,1) #각각의 분류모델 데이터 전처리
   y_test3 = np.array([1 if i == 3 else 0 for i in y_test]).reshape(-1,1)
   
   mylr0 = MyLR([[-1.32069828],
    [-1.02177506],
    [-0.64913889],
    [-0.06329356]]) # The ﬂying cities of Venus (0)
   mylr0.fit_(x_train, y_train0)
   mylr0.alpha = 0.03
   mylr0.fit_(x_train, y_train0)
   mylr0.alpha = 0.3
   mylr0.fit_(x_train, y_train0)
   
   mylr1 = MyLR([[-1.56373886],
    [-0.58824757],
    [ 0.28303058],
    [ 2.20809316]]) #  United Nations of Earth (1)
   mylr1.fit_(x_train, y_train1)
   mylr1.alpha = 0.03
   mylr1.fit_(x_train, y_train1)
   mylr1.alpha = 0.3
   mylr1.fit_(x_train, y_train1)
   
   mylr2 = MyLR([[-2.58616195],
    [ 0.60780971],
    [ 2.8277886 ],
    [ 0.32890994]]) # Mars Republic (2)
   mylr2.fit_(x_train, y_train2)
   mylr2.fit_(x_train, y_train2)
   mylr2.alpha = 0.03
   mylr2.fit_(x_train, y_train2)
   mylr2.alpha = 0.3
   mylr2.fit_(x_train, y_train2)
   
   mylr3 = MyLR([[-4.41035678],
    [ 4.24667587],
    [-3.76787019],
    [-5.23183696]]) # The Asteroids’ Belt colonies (3).
   mylr3.fit_(x_train, y_train3)
   mylr3.alpha = 0.03
   mylr3.fit_(x_train, y_train3)
   mylr3.alpha = 0.3
   mylr3.fit_(x_train, y_train3)
   mylr3.fit_(x_train, y_train3)
   
   print(mylr0.thetas)
   print(mylr1.thetas)
   print(mylr2.thetas)
   print(mylr3.thetas)
   # 모델 생성 완료
   # 전체 데이터 예측
   y_hat0 = mylr0.predict_(x)
   y_hat1 = mylr1.predict_(x)
   y_hat2 = mylr2.predict_(x)
   y_hat3 = mylr3.predict_(x)
   
   
   y_hat_total = np.append(y_hat0, y_hat1, axis=1)
   y_hat_total = np.append(y_hat_total, y_hat2, axis=1)
   y_hat_total = np.append(y_hat_total, y_hat3, axis=1)
   
   y_hat_pre_all = np.array([])
   # 데이터 확률 최댓값을 기준으로 클래스 분류
   for i in range(len(y_hat_total)):
       y_hat_pre_all = np.append(y_hat_pre_all, np.argmax(y_hat_total[i]))
   
   y_hat_pre_all = y_hat_pre_all.reshape(-1,1)
   # 시각화
   y_n = np.array([0.,0.,0.,0.])
   for i in range(len(y)):
       if y[i] == y_hat_pre_all[i]:
           if y[i] == 0:
               y_n[0] += 1
           elif y[i] == 1:
               y_n[1] += 1
           elif y[i] == 2:
               y_n[2] += 1
           elif y[i] == 3:
               y_n[3] += 1
   
   y_n[0] /= np.count_nonzero(y_hat_pre_all == 0)
   y_n[1] /= np.count_nonzero(y_hat_pre_all == 1)
   y_n[2] /= np.count_nonzero(y_hat_pre_all == 2)
   y_n[3] /= np.count_nonzero(y_hat_pre_all == 3)
   plt.bar(range(0, 4), y_n * 100,color=['black', 'red', 'green', 'blue'])
   
   plt.xlabel('class(0,1,2,3)')
   plt.xticks(range(0, 4))
   plt.ylabel('percentage')
   plt.title('**Accuarcy**')
   plt.show()
   
   ```

   <img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg68.png" alt="MLimg0"  />

class2는 80%, class3은 70%의 Accuarcy를 보여준다. 생각보다 정확도가 엄청 높지 않다...

90%이상을 원했지만 쉽지않은 것 같다. 

## Interlude - More Evaluation Metrics!

 cross-entropy로 cost함수를 구현했지만 이외에도 여러가지 cost 함수가 있다.

error를 판단하는 기준은 무엇이 있을까?



```
A single classiﬁcation prediction is either right or wrong, nothing in between. Either an object is assigned to the right class, or to the wrong class. When calculating performance scores for a multiclass classiﬁer, we like to compute a separate score for each class that your classiﬁer learned to discriminate (in a one-vs-all manner). In other words, for a given Class A, we want a score that can answer the question: “how good is the model at assigning A objects to Class A, and at NOT assigning non-A objects to Class A?”

You may not realize it yet, but this question involves measuring two very diﬀerent error types, and the distinction is crucial.
```

A object를 A class에 배정하고 non-A object 를 A class에 배정하지 않는게 굉장히 중요하다고 한다.

### Error Types



#### False positive: when a non-A object is assigned to Class A. 

ex)

- 불이 안났는데 화재경보기를 울리는것
- 그녀가 안아픈데 아픈걸 생각해주는것
- 테디베어에서 face를 식별하는 것



#### False negative: when an A object is assigned to another class than Class A. 

ex)

- 불이 났을때 화재경보기를 안울리는것
- 그녀가 아픈데 안아프다고 생각해주는것 
- 누군가가 있는 이미지에서 얼굴을 인식하지 못하는 것



이 두가지 오류유형을 동시에 최소화하는것은 굉장히 어려운 일이다.

어떻게 사용하는가에 따라서 두가지 오류유형중에 하나를 선택해야 할것이다.

ex) 암을 발견하고자 할때, 건강한 환자에게 암을 진단하는것보다, 암환자에게 암이 아니라고 진단하는게 훨씬 낫다는 것이다.

### Metrics

메트릭기법

**Accuarcy** : 정확한 예측 비율(올바른 클래스가 예측됨), 정확도는 두가지 오류 유형에 대한 정보를 알려주지 않는다.

**Precision**: 어떤 object가 클래스 A에 속한다고 말할때 모델을 얼마나 신뢰할 수 있는지, 즉
클래스 A에 할당된 object의 백분율이며 **False** **Positive**에 대해 제어하려는 경우 precision을 사용하자

**Recall:** 모델이 얼만큼 모든 클래스 A 개체를 인식할 수 있다고 알려준다. 즉 모든 A objects가 Class A로 분류될수 있는가를 백분률로 알려준다.
**False negatives**에 대해 제어하려는 경우 Recall을 사용하자

**F1 score**: **combines precision** and **recall** in **one single measure.** You use the F1 score when want to control **both** **False positives** and **False negatives**



## ex11

목표 : 

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg69.png" alt="MLimg0"  />

위에서 정의한 Accuaracy, precision, recall, f1,을 구현하고 이해하자

```python
import numpy as np

def accuracy_score_(y, y_hat):
    """ Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels y_hat:a numpy.ndarray for the predicted labels
    Returns: The accuracy score as a float. None on any error.
    Raises: This function should not raise any Exception. """
    return np.count_nonzero(y==y_hat) / float(len(y))

def precision_score_(y, y_hat, pos_label=1):
    """ Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int,
    the class on which to report the precision_score (default=1)
    Returns: The precision score as a float.
    None on any error.
    Raises: This function should not raise any Exception. """
    if np.count_nonzero(y_hat == pos_label) == 0:
        return 0
    return np.count_nonzero((y_hat == y) & (y_hat == pos_label)) / float(np.count_nonzero(y_hat==pos_label))

def recall_score_(y, y_hat, pos_label=1):
    """ Compute the recall score.
    Args: y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: The recall score as a float. None on any error.
    Raises: This function should not raise any Exception. """

    return  np.count_nonzero((y_hat == y) & (y_hat == pos_label)) / float(np.count_nonzero((y_hat == y) & (y_hat == pos_label)) + np.count_nonzero((y_hat != y) & (y_hat != pos_label)))

def f1_score_(y, y_hat, pos_label=1):
    """ Compute the f1 score. Args: y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns: The f1 score as a float. None on any error.
    Raises: This function should not raise any Exception"""
    pre =precision_score_(y,y_hat, pos_label=pos_label)
    re = recall_score_(y,y_hat, pos_label=pos_label)
    return 2 * pre * re / (pre + re)
```

numpy의 비교연산, count_nonzero를 이용하여 구현하였다.



## ex12

### Confusion Matrix

목표 

sklearn에 있는 confusion_matrix를 구현해보고 이해하자

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg70.png" alt="MLimg0"  />

**The confusion matrix shows the ways in which your classification model
is confused when it makes predictions.**

추가적으로 pandas의 dafaframe으로 반환하는 보너스 옵션을 만들어도 된다.

```python
# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    confusion_matrix.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/30 18:53:51 by dochoi            #+#    #+#              #
#    Updated: 2020/05/30 20:05:25 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd

def confusion_matrix_(y_true, y_hat, labels=None , df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y_true:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    labels: optional, a list of labels to index the matrix. This may be used to reorder or
    ,! select a subset of labels. (default=None)
    Returns:
    The confusion matrix as a numpy ndarray.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if df_option == True:
        y_actu = pd.Series(y_true)
        y_pred = pd.Series(y_hat)
        df_confusion = pd.crosstab(y_actu, y_pred)
        if labels != None:
            return df_confusion.loc[labels, labels]
        return df_confusion
    else :
        K = max(len(np.unique(y_hat)) , len(np.unique(y)) ) # Number of classes
        name_yhat, index_y_hat = np.unique(y_hat, return_inverse=True)
        name_y, index_y = np.unique(y, return_inverse=True)

        max_y = name_y

        dict = {}
        if(len(name_yhat) > len(max_y)):
            max_y = name_yhat
        for idx, name in enumerate(max_y,):
            dict[name] = idx
        result = np.zeros((K, K), dtype=int)
        for i in range(len(y)):
            result[dict[y[i]]][dict[y_hat[i]]] += 1
        if labels != None:
            result_labels = []
            for elem in max_y:
                if elem not in labels:
                    result_labels = np.append(result_labels, dict[elem])
            arr = np.delete(result, result_labels,axis=0)
            arr = np.delete(arr, result_labels,axis=1)
            return arr
        return result

import numpy as np
from sklearn.metrics import confusion_matrix
y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
# Example 1:
## your implementation
print(confusion_matrix_(y, y_hat))

## sklearn implementation
print(confusion_matrix(y, y_hat))


# # Example 2:
# ## your implementation
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))


## sklearn implementation
print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))

print(confusion_matrix_(y, y_hat,df_option=1))
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet'], df_option=1))
```

구현을 완료하였다 pandas를 이용하는게 numpy보다 훨씬 간단했다.

<img src="https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/Machine%20Learning/image/MLimg71.png" alt="MLimg0"  />

이로써 day03이 완료되었다.



마지막으로 Accuracy, Precision, Recall, F1_score을 다시 한번 정리해보자

**Accuaracy** : 전체분류중에 얼마나 제대로 분류했는가?

**Precision** : 모델이 True로 분류한것 중에 실제로 True인건 얼마인가?

**Recall** : 실제로 True인것 중에 모델이 True로 분류한건 얼마인가?

**F1-score** : 재현율과 정밀도의 조화 평균은 얼마인가?

[깃허브 링크](https://github.com/ChoiDongKyu96/Bootcamp_Machine_Learning)