#include <string>
#include <vector>
#include <algorithm>
using namespace std;

// 전역 변수를 정의할 경우 함수 내에 초기화 코드를 꼭 작성해주세요.
int char_to_int(char c)
{
    if (c == 'A')
        return (0);
    else if (c == 'C')
        return (1);
    else if (c == 'F')
        return (2);
    else if (c == 'J')
        return (3);
    else if (c == 'M')
        return (4);
    else if (c == 'N')
        return (5);
    else if (c == 'R')
        return (6);
    return (7);
}
int solution(int n, vector<string> data) {
       int answer = 0;
    int people[8] ={0,1,2,3,4,5,6,7};
    do{
        int i;
             for(i = 0; i < n;i++)
        {
                 int dist = abs(people[char_to_int(data[i][0])] - people[char_to_int(data[i][2])]) - 1;
                 int comp = data[i][4] - '0';
                 if (data[i][3] == '<')
                 {
                     if (dist >= comp)
                         break;
                 }
                 else if (data[i][3] == '>')
                  {
                     if (dist <= comp)
                         break;
                 }
                 else if (data[i][3] == '=')
                       {
                     if (dist != comp)
                         break;
                 }
        }
        if (i == n)
            answer++;
    }
        while(next_permutation(people, people + 8));
   
 
    return answer;
}
