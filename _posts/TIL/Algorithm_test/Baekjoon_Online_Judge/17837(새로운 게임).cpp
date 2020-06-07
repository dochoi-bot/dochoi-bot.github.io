#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <queue>
#include <memory.h>
#include <algorithm>
using namespace std;
int N, K;
int A[12][12] = { {0} };
int dx[] = { 1,-1,0,0 };
int dy[] = { 0,0,-1,1 };

typedef struct Node {
	int X;
	int Y;
	int D;
	int U;
}Node;
vector<Node*> Nv;
int main(void)
{
	cin.tie(NULL);
	cout.tie(NULL);
	ios_base::sync_with_stdio(false);
//	freopen("input.txt", "r", stdin);

	cin >> N >> K;
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N;j++)
		{
			cin >> A[i][j];
		}
	}
	Node* NNode = new Node;

	Nv.push_back(NNode);
	for (int i = 0; i < K; i++)
	{
		Node* NNode = new Node;
		cin >> NNode->Y >> NNode->X >> NNode->D;
		NNode->D = NNode->D - 1;
		NNode->Y = NNode->Y - 1;
		NNode->X = NNode->X - 1;
		NNode->U = 0;
		Nv.push_back(NNode);
	}
	int cnt = 0;

	while (true)
	{
		cnt++;


		for (int i = 1; i <= K; i++)
		{

			int x = Nv[i]->X;
			int y = Nv[i]->Y;
			int d = Nv[i]->D;
			int nx = x + dx[d];
			int ny = y + dy[d];
			if (nx >= 0 && ny < N && nx < N && ny >= 0)
			{
				if (A[ny][nx] == 0)
				{
					Nv[i]->X = nx;
					Nv[i]->Y = ny;
					int ux = Nv[i]->U;
					for (int j = 1; j <= K; j++)
					{


						if (i == j)
							continue;

						if (Nv[j]->U == i)
						{
							Nv[j]->U = 0;


						}
						if (Nv[j]->X == nx && Nv[j]->Y == ny && Nv[j]->U == 0)
						{
							Nv[j]->U = i;


						}
					}
					while (ux != 0)
					{

						Nv[ux]->X = nx;
						Nv[ux]->Y = ny;

						ux = Nv[ux]->U;

					}

				}
				else if (A[ny][nx] == 1)
				{
					Nv[i]->X = nx;
					Nv[i]->Y = ny;
					int ux = Nv[i]->U;

					//	cout << ux << " ";

					int temp2 = 0;
					for (int j = 1; j <= K; j++)
					{
						if (i == j)
							continue;
						if (Nv[j]->U == i)
						{
							Nv[j]->U = 0;


						}
						if (Nv[j]->X == nx && Nv[j]->Y == ny && Nv[j]->U == 0)
						{
							temp2 = j;
							Nv[j]->U = i;


						}
					}

					while (ux != 0)
					{
						Nv[ux]->X = nx;
						Nv[ux]->Y = ny;
						Nv[temp2]->U = ux;
						ux = Nv[ux]->U;



					}
					ux = Nv[i]->U;
					Nv[i]->U = 0;

					int preidx = i;


					while (ux != 0)
					{

						int temp;
						temp = Nv[ux]->U;
						Nv[ux]->U = preidx;
						preidx = ux;
						ux = temp;


					}

				}
				else if (A[ny][nx] == 2)
				{
					if (d % 2 == 0)
						d = d + 1;
					else
						d = d - 1;
					Nv[i]->D = d;
					nx = x + dx[d];
					ny = y + dy[d];
					if (nx >= 0 && ny < N && nx < N && ny >= 0)
					{
						if (A[ny][nx] != 2)
						{
							if (A[ny][nx] == 1)
							{
								Nv[i]->X = nx;
								Nv[i]->Y = ny;
								int ux = Nv[i]->U;

								//	cout << ux << " ";

								int temp2 = 0;
								for (int j = 1; j <= K; j++)
								{
									if (i == j)
										continue;
									if (Nv[j]->U == i)
									{
										Nv[j]->U = 0;


									}
									if (Nv[j]->X == nx && Nv[j]->Y == ny && Nv[j]->U == 0)
									{
										temp2 = j;
										Nv[j]->U = i;


									}
								}

								while (ux != 0)
								{
									Nv[ux]->X = nx;
									Nv[ux]->Y = ny;
									Nv[temp2]->U = ux;
									ux = Nv[ux]->U;



								}
								ux = Nv[i]->U;
								Nv[i]->U = 0;

								int preidx = i;


								while (ux != 0)
								{

									int temp;
									temp = Nv[ux]->U;
									Nv[ux]->U = preidx;
									preidx = ux;
									ux = temp;


								}
							}
							else
							{
								Nv[i]->X = nx;
								Nv[i]->Y = ny;
								int ux = Nv[i]->U;
								for (int j = 1; j <= K; j++)
								{
									if (i == j)
										continue;
									if (Nv[j]->U == i)
									{
										Nv[j]->U = 0;


									}
									if (Nv[j]->X == nx && Nv[j]->Y == ny && Nv[j]->U == 0)
									{
										Nv[j]->U = i;

									}
								}
								while (ux != 0)
								{
									//cout << "a";
									Nv[ux]->X = nx;
									Nv[ux]->Y = ny;

									ux = Nv[ux]->U;

								}
							}
						}
					}
				}
			}
			else
			{
			
				if (d % 2 == 0)
					d = d + 1;
				else
					d = d - 1;
				Nv[i]->D = d;
				//cout << ny << " " << nx << endl;
				nx = x + dx[d];
				ny = y + dy[d];

				if (nx >= 0 && ny < N && nx < N && ny >= 0)
				{
					if (A[ny][nx] != 2)
					{
						if (A[ny][nx] == 1)
						{
							Nv[i]->X = nx;
							Nv[i]->Y = ny;
							int ux = Nv[i]->U;

							//	cout << ux << " ";

							int temp2 = 0;
							for (int j = 1; j <= K; j++)
							{
								if (i == j)
									continue;
								if (Nv[j]->U == i)
								{
									Nv[j]->U = 0;


								}
								if (Nv[j]->X == nx && Nv[j]->Y == ny && Nv[j]->U == 0)
								{
									temp2 = j;
									Nv[j]->U = i;


								}
							}

							while (ux != 0)
							{
								Nv[ux]->X = nx;
								Nv[ux]->Y = ny;
								Nv[temp2]->U = ux;
								ux = Nv[ux]->U;



							}
							ux = Nv[i]->U;
							Nv[i]->U = 0;

							int preidx = i;


							while (ux != 0)
							{

								int temp;
								temp = Nv[ux]->U;
								Nv[ux]->U = preidx;
								preidx = ux;
								ux = temp;


							}
						}
						else
						{
							Nv[i]->X = nx;
							Nv[i]->Y = ny;

							int ux = Nv[i]->U;
							for (int j = 1; j <= K; j++)
							{
								if (i == j)
									continue;

								if (Nv[j]->U == i)
								{
									Nv[j]->U = 0;

								}
								if (Nv[j]->X == nx && Nv[j]->Y == ny && Nv[j]->U == 0)
								{

									Nv[j]->U = i;
								}
							}
							while (ux != 0)
							{
								//cout << "A";


								Nv[ux]->X = nx;
								Nv[ux]->Y = ny;

								ux = Nv[ux]->U;

							}
						}
					}
				}
			}int unt = 0;
			/*	for (int k = 1; k <= K; k++)
				{
					cout << Nv[k]->U << " [" << Nv[k]->X <<" " << Nv[k]->Y <<"] ";

				}cout << "\n"<< cnt <<endl;*/
				
			for (int k = 1; k <= K; k++)
			{
				unt = Nv[k]->U;
				int nt = 0;
				while (unt != 0)
				{
					nt++;
					unt = Nv[unt]->U;
					if(nt ==3)
					{
						cout << cnt;
						
						return 0;
					}
				}
			}

		

		}
		if (cnt > 1001)
			break;

	}

	cout << "-1";
	return 0;
}
