---
title: 쿠버네티스 입문
layout: post
category: kubernetes
tags:
- Kubernetes
- Docker
excerpt: 쿠버네티스 기초 요약
comments: true
---

# 쿠버네티스 입문

도커가 나타난 이래로 여러가지 오케스트레이션 도구가 등장했다.

경쟁 끝에 쿠버네티스가 정식으로 도커와 통합되었다.

**쿠버네티스란?** 여러 컨테이너를 관리/예약하는 도구

**도커란?** 여러 컨테이너를 관리/예약하는 플랫폼

그런데 둘이 뭐가 다를까?

간단히 얘기해서 도커는 '**기술적인 개념이자 도구**'이고 쿠버네티스는 '**도커를 관리하는 툴**'이라고 생각하면 된다.

이미지를 컨테이너에 띄우고 실행하는 기술이 도커고

이런 도커를 기반으로 컨테이너를 관리하는 서비스가 쿠버네티스라고 생각하면 된다.

도커는 '**한 개의 컨테이너**'를 관리하는 데 최적, 쿠버네티스는 '**여러 개의 컨테이너**'를 서비스 단위로 관리하는 데 최적화되어있다.

## 쿠버네티스란 무엇인가?

**쿠버네티스**는 컨테이너 운영을 자동화 하기 위한 컨테이너 오케스트레이션 도구이고, 구글의 주도로 개발되었다.

**도커 스웜**보다 기능이 풍부하고 생태계가 잘 갖춰져 있다.

도커 호스트 관리, 서버 리소스의 여유를 고려한 컨테이너 배치, 스케일링, 여러 개의 컨테이너 그룹에 대한 로드 밸런싱, 헬스 체크가 등의 기능이 있다.



쿠버네티스는 구글이 컨테이너를 운영하면서 얻은 경험을 바탕으로 만든것이기 대문에 상황에 잘 대응하는 유연성을 갖추고있다.



로컬환경에서 쿠버네티스를 구축하기위해 minikube를 많이 썼지만 쿠버네티스 연동 기능을 이용하면 로컬에서도 쿠버네티스 환경을 구축할 수 있다.

![image-20200816013711431](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200816013711431.png)

![image-20200816014645478](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200816014645478.png)

모두 minikube를 쓰는 이유가 궁금하다...

| 리소스               | 용도                                                         |
| -------------------- | ------------------------------------------------------------ |
| 노드                 | 컨테이너가 배치되는 서버                                     |
| 네임스페이스         | 쿠버네티스 클러스터 안의 가상 클러스터                       |
| 파드                 | 컨테이너의 집합 중 가장 작은 단위로, 컨테인의 실행 방법을 정의한다. |
| 레플리카세트         | 같은 스펙을 갖는 파드를 여러 개 생성하고 관리하는 역할을 한다. |
| 디플로이먼트         | 레플리카 세트의 리비전을 관리한다.                           |
| 서비스               | 파드의 집합에 접근하기 위한 경로를 정의한다.                 |
| 인그레스             | 서비스를 쿠버네티스 클러스터 외부로 노출시킨다.              |
| 컨피그맵             | 설정 정보를 정의하고 파드에 전달한다.                        |
| 퍼시스턴스볼륨       | 파드가 사용할 스토리지 크기 및 종류를 정의.                  |
| 퍼시스턴스볼륨클레임 | 퍼시스턴트 볼륨을 동적으로 확보                              |
| 스토리지클래스       | 퍼시스턴트 볼륨이 확보하는 스토리지의 종류를 정의            |
| 스테이트풀세트       | 같은 스펙으로 모두 동일한 파드를 여러 개 생성하고 관리한다.  |
| 잡                   | 상주 실행을 목적으로 하지 않는 파드를 여러 개 생성하고 정상적인 종료를 보장ㅎ나다. |
| 크론잡               | 크론 문법으로 스케줄링되는 잡                                |
| 시크릿               | 인증 정보 같은 기밀 데이터를 정의한다                        |
| 롤                   | 네임스페이스 안에서 조작 가능한 쿠버네티스 리소스의 규칙을 정의한다. |
| 롤바인딩             | 쿠버네티스 리소스 사용자와 롤을 연결 짓는다.                 |
| 클러스터롤           | 클러스터 전체적으로 조작 가능한 쿠버네티스 리소스의 규칙을 정의한다. |
| 클러스터롤바인딩     | 쿠버네티스 리소스 사용자와 클러스터롤을 연결 짓는다.         |
| 서비스 계정          | 파드가 쿠버네티스 리소스를 조작할 때 사용하는 계정           |



## 쿠버네티스 클러스터와 노드

**쿠버네티스 클러스터**는 쿠버네티스의 여러 리소스를 관리하기 위한 집합체를 말한다.

쿠버네티스 리소스 중에서 가장 큰 개념은 **노드**(**node**)이다

노드는 쿠버네티스 클러스터의 관리 대상으로 등록된 도커 호스트(정확히 말하면 컨테이너의 호스트)로 , 컨테이너가 배치되는 대상이다. 그리고 쿠버네티스 클러스터 전체를 관리하는 서버인 마스터가 적어도 하나 있어야 한다. 쿠버네티스 클러스터는 마스터와 노드의 그룹으로 구성된다.

쿠버네티스는 노드의 리소스 사용 현황 및 배치 전략을 근거로 컨테이너를 적절히 배치한다.

클러스터의 처리 능력은 노드에 의해 결정된다.

![image-20200825182936557](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825182936557.png)



## 네임스페이스

쿠버네티스는 클러스터 안에 가상 클러스터를 또 다시 만들 수 있다.

이 클러스터 안의 가상 클러스터를 namespace라고 한다. 클러스터를 처음 구축하면 default, docker, kube-public, kube-system의 네임스페이스 4개가 만들어져 있다.

![image-20200825184119412](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825184119412.png)

네임스페이스는 개발팀이 일정 규모 이상일때 유용하다.(각자의 권한을 설정하여 개발을 효율적으로 한다.)

## 파드

파드(pod)는 컨테이너가 모인 집합체의 단위로, 적어도 하나 이상의 컨테이너로 이루어져 있다.

쿠버네티스를 도커와 함께 사용한다면 파드는 컨테이너 하나 혹은 컨테이너의 집합체가 된다.

쿠버네티스에서는결합이 강한 컨테이너를 파드로 묶어 일괄 배포한다. 컨테이너가 하나일 경우에도 파드로 배포한다.

파드는 파드하나가 여러 노드에 걸쳐 배치될 수는 없지만, 같은 파드를 여러 노드에 배치할 수 있고, 한 노드에 여러개 배치할 수도 있다.

파드 생성은 kubectl만 새아요해도 가능하지만 yaml파일로 정의하는것이 좋다. 쿠버네티스의 여러가지 리소스를 정의하는 파일을 매니페스트 파일이라고 한다.



Pod를 정의하는 yaml파일을 만들어보자

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: simple-echo
spec:
  containers:
  - name: nginx
    image: gihyodocker/nginx:latest
    env:
    - name: BACKEND_HOST
      value: localhost:8080
    ports:
    - containerPort: 80
  - name: echo
    image: gihyodocker/echo:latest
    ports:
    - containerPort: 8080

```

```
simple-pod.yaml
```



```
kubectl exec -it simple-echo sh -c nginx
```

```
kubectl logs -f simple-echo -c echo
```

![image-20200825190214846](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825190214846.png)

![image-20200825190221782](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825190221782.png)

-c 옵션은 파드안에 컨테이너가 여러개일경우 컨테이너를 지정하는 옵션이다

```
kubectl delete pod simple-echo
```

파드를 삭제하는 명령어

```
kubectl delete -f simple-pod.yaml
```

yaml파일 이름으로도 파드를 삭제할 수 있다. 이 방법을 사용하면 매니페스트에 작성된 리소스 모두가 삭제된다.

![image-20200825192505836](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825192505836.png)

파드를 정의한 매니페스트 파일로는 파드를 하나밖에 생성할 수 없다. 그러나 어느정도 규모가 되는 애플리케이션을 구축하려면 같은 파드를 여러 개 실행해 가용성을 확보해야 하는 경우가 생긴다.

이런 경우에 사용하는것이 레플리카세트이다.

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: echo
  labels:
    app: echo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: echo
  template: # template 아래는 파드 리소스 정의와 같음
    metadata:
      labels:
        app: echo
    spec:
      containers:
      - name: nginx
        image: gihyodocker/nginx:latest
        env:
        - name: BACKEND_HOST
          value: localhost:8080
        ports:
        - containerPort: 80
      - name: echo
        image: gihyodocker/echo:latest
        ports:
        - containerPort: 8080

```

![image-20200825193408080](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825193408080.png)

3개가 생긴것을 확인할 수 있다.

## 디플로이먼트

레플리카세트보다 상위에 해당하는 리소스로 디플로이먼트가 있다. 디플로이먼트는 애플리케이션 deploy의 기본 단위가 되는 리소스이다. 레플리카세트가 똑같은 파드의 개수를 관리한다면, 디플로이먼트는 레플리카세트를 관리하고 다룬다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: echo
  labels:
    app: echo
spec:
  replicas: 3
  selector:
    matchLabels:
      app: echo
  template: # template 아래는 파드 리소스 정의와 같음
    metadata:
      labels:
        app: echo
    spec:
      containers:
      - name: nginx
        image: gihyodocker/nginx:latest
        env:
        - name: BACKEND_HOST
          value: localhost:8080
        ports:
        - containerPort: 80
      - name: echo
        image: gihyodocker/echo:patched
        env:
        - name: HOGE
          value: fuga
        ports:
        - containerPort: 8080

```

![image-20200825194153395](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825194153395.png)

디플로이먼트는 물론이고, 레플리카와 파드까지 생성이 되었다.

![image-20200825194613096](https://raw.githubusercontent.com/dochoi-bot/TIL/master/Kubernetes/images/image-20200825194613096.png)

## 레플리카세트의 생애주기

쿠버네티스는 디플로이먼트를 단위로 애플리케이션을 배포한다. 실제 운영에서는 레플리카세트를 직접 다루기보다는 디플로이먼트 매니페스트파일을 통해 다루는 경우가 대부분이다.

따라서 레플릴카세트 동작 방식을 파악하자

### 파드개수만 수정하면 레플리카세트가 새로 생성되지 않음

image를 수정해야 새로 생성된다.(리비전이 바뀜)





## 롤백 실행하기

디플로이먼트는 리비전 번호가 기록되므로 특정 리비전의 내용을 확인할 수 있다.

```
kubectl rollout history deployment echo --revision=1
```

**참고**

도커/쿠버네티스를 활용한 컨테이너 개발 실전 입문 현장에서 바로 활용할 수 있는 컨테이너 개발 기법과 실전 기술
야마다 아키노리 저/심효섭 역 - 위키북스