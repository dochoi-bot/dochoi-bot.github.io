---
title: "[iOS] AutoLayout 프로그래밍 방식, 코드로 적용하기"
layout: post
category: ios
tags:
- iOS
excerpt: "[iOS] AutoLayout 프로그래밍 방식, 코드로 적용하기"
comments: true
---

# [iOS] AutoLayout 프로그래밍 방식, 코드로 적용하기

`autoLayout`을 프로그래밍 방식으로 적용하기에 앞서 기초지식을 학습해야합니다.

`autoLayout`을 위해 `StackView`를 사용하였습니다.



**stackView 옵션들**

`translatesAutoresizingMaskIntoConstraints` 

- 오토레이아웃을 시스템이 정할것인가? 기본은 true이다. 저는 제가 직접 constraint를 주어서 비율에 맞춰 변하게 하고싶기 때문에

  이 값을 false로 하였습니다.

`distribution `

- fill

  - 공간이 유효하면 이미지를 view's axis를 따라서 배치합니다. 정렬된 뷰가 Stackview에 맞지 않을경우 우선순위에 따라 이미지를 줄이거나 늘립니다.
  - ![image-20200903155122022](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/iOS/images/image-20200903155122022.png)

- fillEqually

  - fill과 같지만 이미지를 건드리지 않습니다.
  - ![img](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/iOS/images/img.png)

- fillProportionally

  - fill과 같지만 이미지가 intrinsic content size에 따라 비례적으로 적용됩니다.
  - intrinsic ContentSize =  컨텐츠 고유 크기

  ![image-20200903155122022](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/iOS/images/image-20200903155122022.png)

- equalSpacing

  - 뷰간 간격을 둡니다.
  - ![image-20200903155154870](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/iOS/images/image-20200903155154870.png)

- equalCentering

  - 뷰의 중심간 간격을 둔다.
  - ![image-20200903155213553](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/iOS/images/image-20200903155213553.png)


**오토레이아웃에서 자주쓰는 Anchor** : component를 다른 componet에 고정하기위한 객체 



- axis는 StackView의 가로, 세로 형태를 설정합니다.
- alignment는 StackView안의 view들이 Y축 정렬을 설정합니다.
- distribution은 StackView안의 view들이 X축 정렬을 설정합니다.
- spacing은 view들간의 간격을 설정합니다.

![img](https://raw.githubusercontent.com/ChoiDongKyu96/TIL/master/iOS/images/stackViewComponents2.png)

#### **프로그래밍 방식으로 오토 레이아웃을 할때 유의사항**

- Constraints를 활성화시키기 전에  프로그래밍방식으로 만든 뷰를 상위 뷰에 추가했는가?
- 제약조건 코드는 **viewDidLoad() / viewWillAppear()**에 쓰면 안된다. **updateViewConstraints or viewWillLayoutSubviews** 이곳에 써야한다. (뷰가 로드되기 전에 적용해야합니다. )
- **translatesAutoresizingMaskIntoConstraints**를 OFF했는지 확인하라



#### **leadingAnchor 와 trailingAnchor, Left , Right 의 차이점**

leadingAnchor  :  A layout anchor representing the leading edge of the view’s frame.

trailingAnchor : A layout anchor representing the trailing edge of the view’s frame.

left, right 와 다른점은 언어에 따라 방향이바뀝니다.(아랍어 등) 애플은 leadingAnchor, trailingAnchor사용을 권장합니다.



**View에서 알면 좋은것들**

[`case scaleToFill`](https://developer.apple.com/documentation/uikit/uiview/contentmode/scaletofill)

The option to scale the content to fit the size of itself by changing the aspect ratio of the content if necessary.

[`case scaleAspectFit`](https://developer.apple.com/documentation/uikit/uiview/contentmode/scaleaspectfit)

The option to scale the content to fit the size of the view by maintaining the aspect ratio. Any remaining area of the view’s bounds is transparent.

[`case scaleAspectFill`](https://developer.apple.com/documentation/uikit/uiview/contentmode/scaleaspectfill)

The option to scale the content to fill the size of the view. Some portion of the content may be clipped to fill the view’s bounds.



Example

```swift
 private let stackView: UIStackView =
    {
        let stackViewTemp = UIStackView()
        stackViewTemp.translatesAutoresizingMaskIntoConstraints = false
        stackViewTemp.distribution = .fillEqually
        stackViewTemp.spacing = 7
        return stackViewTemp
    }()
```

```swift
    func setStackViewAutolayout() -> Void
    {
        self.stackView.topAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.topAnchor).isActive = true
        self.stackView.leadingAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.leadingAnchor, constant: 7).isActive = true
        self.stackView.trailingAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.trailingAnchor, constant: -7).isActive = true
    }
```