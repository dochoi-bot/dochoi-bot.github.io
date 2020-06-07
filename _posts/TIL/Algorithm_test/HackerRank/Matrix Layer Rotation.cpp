#include <bits/stdc++.h>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);
vector<string> split(const string &);

// Complete the matrixRotation function below.
void insert(int *temp, vector<vector<int> > &matrix, int cnt)
{

    int base_idx = 0;
    int m = matrix[0].size();
    int n = matrix.size();
    for(int i = 0 + cnt ; i < n - cnt; i++)
    {
        matrix[i][cnt] = temp[base_idx];
          base_idx++;
    }
    for(int i = 1 + cnt ; i < m - cnt; i++)
    {
            matrix[n - cnt - 1][i] = temp[base_idx];
          base_idx++;
    }
    for(int i = n - cnt - 2 ; i >= cnt; i--)
    {

            matrix[i][m - cnt - 1] = temp[base_idx];
           base_idx++;
    }
    for(int i = m - cnt - 2; i >= cnt + 1; i--)
    {
            matrix[cnt][i] = temp[base_idx];
          base_idx++;
    }
}
int gofind(vector<vector<int> > &matrix, int r, int cnt, int check[301][301])
{
    int temp[1200]= {0};
    int base_idx = 0;
    int one_rotate = 2 * (matrix.size() - (2 * cnt) + matrix[0].size() - (2 * cnt )) - 4;   int m = matrix[0].size();
    int n = matrix.size();
    for(int i = 0 + cnt ; i < n - cnt; i++)
    { 

        if (check[i][cnt] == 0)
            check[i][cnt] = 1;
      
        int convert_idx = (base_idx + r) % one_rotate;
        temp[convert_idx] = matrix[i][cnt];
          base_idx++;
    }   
    for(int i = 1 + cnt ; i < m - cnt; i++)
    {
        if (check[n - cnt - 1][i] == 0)
            check[n - cnt - 1][i] = 1;
      
        int convert_idx = (base_idx + r) % one_rotate;
        temp[convert_idx] = matrix[n - cnt - 1][i];
          base_idx++;
    }
    for(int i = n - cnt - 2 ; i >= cnt; i--)
    {
        if (check[i][m - cnt - 1] == 0)
            check[i][m - cnt - 1] = 1;
        int convert_idx = (base_idx + r) % one_rotate;
        temp[convert_idx] = matrix[i][m - cnt - 1];
          base_idx++;
    } 
    for(int i = m - cnt - 2; i >= cnt + 1; i--)
    {
        if (check[cnt][i] == 0)
            check[cnt][i] = 1;
        int convert_idx = (base_idx + r) % one_rotate;
        temp[convert_idx] = matrix[cnt][i];
          base_idx++;
    }

    insert(temp, matrix, cnt);

    if (check[cnt + 1][cnt + 1] == 0)
        return (1);
    return (0);
}
void matrixRotation(vector<vector<int>> matrix, int r) {
    int check[301][301] = {{0}};
    int cnt = 0;
    while (gofind(matrix, r, cnt, check))
    {
        cnt++;
    }
    for (int i  = 0 ; i < matrix.size(); i++)
    {
        for(int j = 0 ; j < matrix[0].size(); j++)
        {
            cout << matrix[i][j] <<" ";
        }cout << "\n";
    }
}

int main()
{
    string mnr_temp;
    getline(cin, mnr_temp);

    vector<string> mnr = split(rtrim(mnr_temp));

    int m = stoi(mnr[0]);

    int n = stoi(mnr[1]);

    int r = stoi(mnr[2]);

    vector<vector<int>> matrix(m);

    for (int i = 0; i < m; i++) {
        matrix[i].resize(n);

        string matrix_row_temp_temp;
        getline(cin, matrix_row_temp_temp);

        vector<string> matrix_row_temp = split(rtrim(matrix_row_temp_temp));

        for (int j = 0; j < n; j++) {
            int matrix_row_item = stoi(matrix_row_temp[j]);

            matrix[i][j] = matrix_row_item;
        }
    }

    matrixRotation(matrix, r);

    return 0;
}

string ltrim(const string &str) {
    string s(str);

    s.erase(
        s.begin(),
        find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace)))
    );

    return s;
}

string rtrim(const string &str) {
    string s(str);

    s.erase(
        find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(),
        s.end()
    );

    return s;
}

vector<string> split(const string &str) {
    vector<string> tokens;

    string::size_type start = 0;
    string::size_type end = 0;

    while ((end = str.find(" ", start)) != string::npos) {
        tokens.push_back(str.substr(start, end - start));

        start = end + 1;
    }

    tokens.push_back(str.substr(start));

    return tokens;
}
