#define _CRT_SECURE_NO_WARNINGS
#include <string>
#include <vector>
#include <queue>
#include <memory.h>
#include <iostream>
using namespace std;
typedef struct Node {
	int idx;

	int num;

}Node;
int A[10] = { 0 };
int answer = 0;
bool C[33] = { 0 };
vector<Node*> v;

int S0[] = {0, 2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,0 };
int S1[] = {0, 13,16,19};
int S2[] = { 0,22 ,24,25,30,35};
int S3[] = { 0,28,27,26 };
int  covertoidx(int num, int pos)
{
	if (num == 0)
	{
		
		return pos;


	}
	else if (num == 1)
	{
		if (pos == 1)
		{
			return 21;
		}
		if (pos == 2)
		{
			return 22;
		}
		if (pos == 3)
		{
			return 23;
		}
		
	}
	else if (num == 2)
	{
		if (pos == 1)
		{
			
				return 24;
			
		}
		if (pos == 2)
		{
			
				return 25;
			
		}
		if (pos == 3)
		{
		
				return 29;
		
		}if (pos == 4)
		{
		
				return 30;
	
		}if (pos == 5)
		{
		
				return 31;
		
		}
		
	}
	else if (num == 3)
	{
		if (pos == 1)
		{
		
			return 26;
		
		}
		if (pos == 2)
		{
		
				return 27;
			
		}
		if (pos == 3)
		{
			
				return 28;
		
		}
		
	}
}
int move(int _idx, int cnt)
{
	int add = 0;
	int cur_pos = v[_idx]->idx;
	//int cnt = v[_idx]->cnt;
	int num = v[_idx]->num;
	if (cur_pos == 5 && num == 0)
	{
		num = 1;
		cur_pos = 0;
	}
	else if (cur_pos == 10 && num == 0)
	{
		num = 2;
		cur_pos = 0;
	}
	else if (cur_pos == 15 && num == 0)
	{
		num = 3;
		cur_pos = 0;
	}
	int pos = cur_pos;
	for (int i = 0; i < cnt; i++)
	{
		if (num == 0)
		{
			pos += 1;
			if (pos < 21)
			{
				;
			}
			else
			{
				pos = 32;
				break;
			}

		}
		else if (num == 1)
		{
			pos += 1;
			if (pos < 4)
			{
				;
			}
			else
			{
				num = 2;
			
				pos = 3;
			}
		}
		else if (num == 2)
		{
			pos += 1;
			if (pos < 6)
			{
				;
			}
			else
			{
				num = 0;
				
				pos = 20;
				
			}
		}
		else if (num == 3)
		{
			pos += 1;
			if (pos < 4)
			{
				;
			}
			else
			{
				num = 2;
				
				pos = 3;

			}
		}
	}
	if (num == 0)
	{
		if (pos < 21)
		{
			if (C[pos] == 1)
				return -1;
			C[pos] = 1;
		}





	}
	else if (num == 1)
	{
		if (pos == 1)
		{
			if (C[21] == 1)
				return -1;

			C[21] = 1;


		}
		if (pos == 2)
		{
			if (C[22] == 1)
				return -1;

			C[22] = 1;

		}
		if (pos == 3)
		{
			if (C[23] == 1)
				return -1;

			C[23] = 1;

		}

	}
	else if (num == 2)
	{
		if (pos == 1)
		{
			if (C[24] == 1)
				return -1;

			C[24] = 1;

		}
		if (pos == 2)
		{
			if (C[25] == 1)
				return -1;

			C[25] = 1;

		}
		if (pos == 3)
		{
			if (C[29] == 1)
				return -1;

			C[29] = 1;

		}if (pos == 4)
		{
			if (C[30] == 1)
				return -1;

			C[30] = 1;

		}if (pos == 5)
		{
			if (C[31] == 1)
				return -1;

			C[31] = 1;

		}
	
	}
	else if (num == 3)
	{
		if (pos == 1)
		{
			if (C[26] == 1)
				return -1;

			C[26] = 1;

		}
		if (pos == 2)
		{
			if (C[27] == 1)
				return -1;

			C[27] = 1;

		}
		if (pos == 3)
		{
			if (C[28] == 1)
				return -1;

			C[28] = 1;

		}

	}
	if (pos == 32)
	{
		int pos2 = v[_idx]->idx;
		int num2 = v[_idx]->num;
		int iidx = covertoidx(num2, pos2);
		C[iidx] = 0;
	v[_idx]->idx = pos;
	v[_idx]->num = num;

	return 0;
}
		int pos2 = v[_idx]->idx;
		int num2 = v[_idx]->num;
		int iidx = covertoidx(num2, pos2);
		C[iidx] = 0;
		v[_idx]->idx = pos;
		v[_idx]->num = num;
		if (num == 0)
		{
			add = S0[pos];
		}
		else if (num == 1)
		{
			add = S1[pos];
		}
		else if ( num ==2)
		{
			add = S2[pos];
		}
		else if ( num==3)
		{
			add = S3[pos];
		}
		//cout << add << " ";
		return add;
	
}
void find(int _cnt, int _sum)
{
	int sum = _sum;
	
	//cout << sum <<" "; 
		
		if (_cnt == 10)
		{
			
			answer = max(answer, sum);
			//cout << endl;
			return;
		}
			
	for (int i = 0; i < 4; i++)
	{	
		int temp = v[i]->idx;
		int temp2 = v[i]->num;
		if (temp == 32)
			continue;
		int C2[32] = { 0 };
		for (int j = 0; j < 32; j++)
		{
			C2[j] = C[j];
		}
		 int add = move(i , A[_cnt]);
		
		 if (add == -1)
		 {
		
			 continue;
		 }

	
		 
		 find(_cnt + 1, sum +add);
		 v[i]->num = temp2;
		 v[i]->idx = temp;
		 for (int j = 0; j < 32; j++)
		 {
			 C[j] = C2[j];
		 }




		

	}

}
int main()
{
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);
	//freopen("input.txt", "r", stdin);
	
	answer = 0;
	for (int i = 0; i < 10; i++)
	{
		if (i < 4)
		{
			Node* NNode = new Node;
			NNode->idx = 0;
		
			NNode->num = 0;
			v.push_back(NNode);

		}
		cin >> A[i];
	}
	find(0, 0);
	cout << answer;
}
