---
layout: post
title: "2. 문제 해결 설계"
excerpt: "문제 해결 설계"
date: 2020-06-08
category : algorithm
tags: [Algorithm]
comments: true
---

# 문제 해결 설계



## 도입

프로그래밍 대회는 문제 해결 능력을 수련하기에 무척 좋은 환경이다.

하지만 **무작정 알고리즘을 외우고 문제를 푼다고 해서 문제 해결 실력이 쌓이는 것은 아니다.**

```
스스로 끊임없이 문제를 어떤 방식으로 해결하는 지 의식하고, 어느 부분이 부족한지, 어떤 부분을 개선해야 하는지 파악해야 한다.
```



## 문제 해결 과정

양자역학의 어려운 문제를 푼 알고리즘 "파인만 알고리즘"

```
1. 문제를 적는다.
2. 골똘이 생각한다.
3. 답을 적는다.
```



별 의미가 없는 것 같지만 이는 문제 해결 과정의 기본이다.

이를 좀 더 구체화 해보자

```
1. 문제를 읽고 이해한다.
2. 문제를 익숙한 용어로 재정의한다.
3. 어떻게 해결할지 계획을 세운다.
4. 계획을 검증한다.
5. 프로그램으로 구현한다.
6. 어떻게 풀었는지 돌아보고 ,개선할 방법이 있는지 찾아본다.
```



### 1단계 문제를 읽고 이해하기

조급한 마음에 입출력 예제만 보고 문제가 원하는 것을 **유추하기 십상**이다. 그러나 이와 같은 성급한 행동은 언젠가 대가를 치룬다. 문제의 궁극적인 목적을 옳게 이해하더라도 **사소한 제약조건**을 잘못 이해하면 풀 수 없는 문제가 존재한다.



### 2단계 재정의와 추상화

문제를 자신의 언어로 풀어씀으로써 문제가 요구하는 바를 직관적으로 이해할 수 있다.
이는 요구하는 바가 복잡한 문제일수록 꼭 해야 하는 단계이다.



### 3단계 계획세우기

문제를 어떤 방식으로 해결할지 결정하고, 사용할 알고리즘과 자료구조를 선택한다.



### 4단계 계획 검증하기

설계한 알고리즘이 모둔 경우에 요구 조건을 정확히 수행하는지 증명하고, 수행에 걸리는 시관과 사용하는 메모리가 문제의 제한 내에 들어가는지 확인한다.



### 5단계 계획 수행하기

아무리 천재적인 알고리즘을 고안하더라고 구현이 부정확하면 프로그램은 돌아가지 않는다.
따라서 구현과정의 중요성은 정말 중요하다.



### 6단계 회고하기

자신이 문제를 해결한 과정을 돌이켜보고 계산하는 과정을 말한다.
이 책에선 이 6단계를 굉장히 강조한다. **오답 원인, 접근 방식, 깨달음, 다른사람의 코드 보기** 등 으로
깨달음을 얻을 수 있다고 한다.



### 문제를 풀지 못할 때

일정 시간이 지나도 답이 안나올 때는 다른사람의 소스코드를 참조하자



### 주의

이 문제해결과정은 어디까지나 이론일 뿐 자신에게 맞출 필요가 있다.



## 문제 해결 전략

### 직관과 체계적인 접근

- **비슷한 문제를 풀었던가?**
- **단순한 방법에서 시작할 수 있을까?**
  - 시간과 공간 제약을 생각하지 않고 일단 가장 단순한 알고리즘을 만들어보자
  - 간단히 풀 수 있는 문제를 너무 복잡하게 생각하여 어렵게 푸는 실수를 방지할 수 있다.

- **내가 문제를 푸는 과정을 수식화할 수 있을까?**
  
  - 점진적인 접근방식은 아니지만 손으로 몇가지 입력을 해결한후 공식화 할 수 있는지 확인해 보는것도 괜찮다.
- **문제를 단순화할 수 없을까?**
  - 문제의 제약조건을 없앨 수 있을까?
  - 계산해야하는 변수의 수를 줄일 수는 없을까?
  - 다차원의 문제를 1차원으로 이동시킬 수 있을까?
  - ex)42서울에서 진행했던 nqueen문제가 있다.
- **그림으로 그려볼 수 있을까?**
  
  - 사람의 사고 체계는 숫자의 나열보다 기하학적 도형을 더 직관적으로 받아들이기 때문에 이런 접근방식도 괜찮다.
- **수식으로 표현할 수 있을까?**
  - 평문으로 쓰여 있는 문제를 수식으로 표현하는것이 도움이 될 경우가 있다.
  - 순수한 수학적 조작이 문제를 해결하는 데 큰 도움을 줄 수 도 있다.
- **문제를 분해할 수 있을까?**
  
  - 문제를 더 다루기 쉬운 형태로 변형할 수 있을까?
- **뒤에서부터 생각해서 문제를 풀 수 있을까?**
  
- ex) 사다리게임
  
- **순서를 강제할 수 있을까?**

  - bfs , dfs를 말하는것 같다.

- **특정 형태의 답만을 고려할 수 있을까?**

  