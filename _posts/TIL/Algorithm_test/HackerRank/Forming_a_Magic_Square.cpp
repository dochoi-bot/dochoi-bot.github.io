#include <bits/stdc++.h>

using namespace std;

// Complete the formingMagicSquare function below.


int find_cost(int temp[3][3], vector<vector <int> > &s)
{
    int sum  =0 ;
    for(int i = 0 ;i  <3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            sum += abs(s[i][j] - temp[i][j]);
        }
    }
    return (sum);
}
void insert_temp(int x, int y, int temp[3][3],int *answer, vector<vector<int>> &s, int check[10])
{
    if (y == 3)
    {
        int f_sum = temp[0][0] + temp[0][1] + temp[0][2];

    int sum_cross = temp[0][0] + temp[1][1] + temp[2][2];
    int sum_cross2 = temp[2][0] + temp[1][1] + temp[0][2];
    int sum_3col = temp[0][2] + temp[1][2] + temp[2][2];
         if (sum_cross != f_sum || sum_cross2 != f_sum || sum_3col != f_sum)
            return;

        *answer=  min(*answer, find_cost(temp, s));
        return;
    }
    else if (y == 2 && x == 0)
    {
        int sum_row0 = temp[0][0] + temp[0][1] + temp[0][2];
    int sum_row1 = temp[1][0] + temp[1][1] + temp[1][2];
    if (sum_row0 != sum_row1)
        return ;
    }
    else if (y == 2 && x == 1)
    {
        int sum_row0 = temp[0][0] + temp[0][1] + temp[0][2];
    int sum_col0 = temp[0][0] + temp[1][0] + temp[2][0];
    if (sum_row0 != sum_col0)
        return ;
    }
    else if (y == 2 && x == 2)
    {
            int sum_row0 = temp[0][0] + temp[0][1] + temp[0][2];
        int sum_col1 = temp[0][1] + temp[1][1] + temp[2][1];
        if (sum_row0 != sum_col1)
            return ;
    }
    for(int i = 1; i <= 9; i++)
    {
        if (check[i] == 1)
            continue;
        temp[y][x]= i;
        check[i] = 1;
        if (x == 2)
            insert_temp(0, y + 1, temp, answer, s, check);
        else
            insert_temp(x + 1, y , temp, answer, s, check);
        check[i] =0;
    }
}
int formingMagicSquare(vector<vector<int>> s) {
    int temp[3][3] = {1};
    int answer = 99999;
    int check[10] = {0};
    insert_temp(0 , 0, temp, &answer, s, check);
    return answer;
}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    vector<vector<int>> s(3);
    for (int i = 0; i < 3; i++) {
        s[i].resize(3);

        for (int j = 0; j < 3; j++) {
            cin >> s[i][j];
        }

        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    int result = formingMagicSquare(s);

    fout << result << "\n";

    fout.close();

    return 0;
}
