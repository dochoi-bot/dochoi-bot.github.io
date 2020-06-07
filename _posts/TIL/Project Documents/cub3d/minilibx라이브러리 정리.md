# minilibx 라이브러리 함수들과 기능을 정리한다.



## MiniLibX - Simple Graphical Interface Library for students

```c
#include  <mlx.h>

void  *
mlx_init();
```



### DESCRIPTION

minilibx는 그래픽 소프트웨어를 쉽게 만들 수 있다. X window나 Cocoa 프로그래밍 지식 없이, 이는 간단한 창 생성 그리기 도구, 이미지를 다루는 기능을 제공한다.

X-Window는 Unix를 위한 network-oriented graphical system이다.

2가지로 기능을 나눌 수 있다.

소프트 웨어가 화면에 무언가를 그리거나, 키보드와 마우스의 입력을 가져오는것

다른쪽은 X-server가 화면, 키보드, 마우스를 관리한다.(디스플레이라 부르기도 한다.)

소프트웨어와 X-server의 네트워크 연결이 전송되어야 한다.

### MACOS CONCEPT

간단히 말하면 키보드와 마우스의 입력을 받는거 따로, 그래픽을 그리고, 입력받은것을 처리하는거 따로 있는데 이 두가지를 연결해야한다.

### LIBRARY FUNCTIONS

1, 소프트웨어와 디스플레이의 연결을 초기화 해야한다.

이 연결이 설정되면 "이 창에 노란색 픽셀을 그리고 싶다" 또는 "사용자가 키를 쳤는가?"와 같은 그래픽 오더를 보낼 수 있는 다른 MiniLibX 기능을 사용할 수 있다.

mlx_init()함수가 이기능을 한다. 매개변수가 필요없으며 나중에 필요한 void*주소를 반환한다.



### RETURN VALUES

연결에 에러가 나면 NULL을 반환하고 그렇지 않으면 NULL이 아닌 포인터가 연결 식별자로 반환된다.



## MiniLibX - Managing windows

```C
void *
mlx_new_window ( void *mlx_ptr, int size_x, int size_y, char *title );

int
mlx_clear_window ( void *mlx_ptr, void *win_ptr );

int
mlx_destroy_window ( void *mlx_ptr, void *win_ptr );
```

### DESCRIPTION

mlx_new_window()함수는 sizex와 size_y크기의 title이라는 제목의 새창을 만든다. 

mlx_ptr은 mlx_init()에 의해 반환된 연결 식별자이다.(void*)

mlx_new_winodw()는 다른 MiniLibx에서 사용할 수 있는 void*창 식별자를 반환한다.

이는 임의 개수의 개별창을 처리할 수 있다.



mlx_clear_window() 그리고 mlx_destory_window()는 각각 clear(블랙으로 만듬)

그리고 창을 없앤다. 매개변수는 mlx_ptr, new_winodw로 만든 식별자이다.

### RETURN VALUES

mlx_new_window는 새 창을 생성하는것에 실패하면 NULL이 반환되고 그렇지 않으면 ono-null pointer인 window identifier이 반환된다.

mlx_clear_window와 mlx_destory_window는 당장은 아무것도 리턴하지 않는다.



## MiniLibX - Drawing inside windows

```c
int
mlx_pixel_put ( void *mlx_ptr, void *win_ptr, int x, int y, int color );

int
mlx_string_put ( void *mlx_ptr, void *win_ptr, int x, int y, int color, char *string );
```

### DESCRIPTION

mlx_pixel_put()function은  win_ptr 창 위에 (x, y)좌표에 정의된 픽셀을그린다(특정 색을)

왼쪽 위를 0.0으로 하는 좌표계를 쓴다. (우하 좌표계) 창의 포인터인, mlx_ptr이 필요하다.

mlx_string_put()도 유사한 함수다. string을 윈도우에 display한다.

두가지 함수에서  창을 벗어난 곳에는 어떤것도 표시 할 수 없다.

색은 RGB, (0~255)의 값으로 구분된다.( 인디언 문제를 주의해라) 

(0 R G B) 총 4바이트의 int값