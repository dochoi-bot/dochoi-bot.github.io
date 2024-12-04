---
layout: post
category: ios
tags:
- iOS
excerpt: "[iOS]  UIStatusBarStyle 설정 방법"
comments: true
title: "[iOS] UIStatusBarStyle 설정 방법"
---

# [iOS] UIStatusBarStyle 설정 방법

iOS를 만들다보면 UIStatusBar Style을 설정해야 할 때가 있습니다.

Status bar style은 3개가 있습니다.

| Status bar style | Value                        | Status bar text color                                        |
| :--------------- | :--------------------------- | :----------------------------------------------------------- |
| Default          | UIStatusBarStyleDefault      | 유저의 인터페이스 스타일에따라 자동으로 설정합니다. Black text when `traitCollection.userInterfaceStyle` = `UIUserInterfaceStyle.light` and white text when `traitCollection.userInterfaceStyle` = `UIUserInterfaceStyle.dark` |
| Dark Content     | UIStatusBarStyleDarkContent  | 밝은 배경을 고려한 검정색 텍스트                             |
| Light Content    | UIStatusBarStyleLightContent | 어두운 배경을 고려한 하얀색 텍스트                           |

![image-20200902205115136](https://raw.githubusercontent.com/dochoi-bot/TIL/master/iOS/images/image-20200902205115136.png)

왼쪽이 LightContent, 오른쪽이 Dark Content입니다.

Status Bar 설정은 크게 2가지 방법이 있습니다.



## 첫번째 방법(모든 뷰 적용, plist수정)

![image-20200905013401572](https://raw.githubusercontent.com/dochoi-bot/TIL/master/iOS/images/image-20200905013401572.png)

설정에서 Status Bar 설정을 할 수 있으나 빌드시켜보면 적용이 안되는 것을 알 수 있습니다.

이유는  `View controller-based status bar appearance` 이 친구 입니다.

이 친구의 기본값은 True로, False로 바꿔줘야 설정이 됩니다. False로 추가해주면 적용이 되는것을 확인할 수 있습니다.



## 두번째 방법(각 뷰 적용, 프로그래밍 방식)

간단합니다

```swift
   override var preferredStatusBarStyle: UIStatusBarStyle {
          return .lightContent
    }
```

원하는 뷰 컨트롤러에서 이 코드를 넣어주면 됩니다. 

 `View controller-based status bar appearance`이 친구와 전혀 상관없이 적용이 됩니다.