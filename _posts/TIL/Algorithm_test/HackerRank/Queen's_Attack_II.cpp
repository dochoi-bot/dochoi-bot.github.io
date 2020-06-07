#include <bits/stdc++.h>

using namespace std;

vector<string> split_string(string);


// Complete the queensAttack function below.
int queensAttack(int n, int k, int r_q, int c_q, vector<vector<int>> obstacles) {
    unordered_set<int> s[100000];
    for(int i = 0; i  < obstacles.size(); i++)
{
    s[obstacles[i][0] - 1].insert(obstacles[i][1] - 1);
}
r_q--;
c_q--;
int answer = 0;
for(int i = r_q - 1; i >= 0; i-- )
{
    if (s[i].end() != s[i].find(c_q))
        break;
    answer++;
}
for(int i = r_q - 1, j = c_q + 1; i >= 0 && j < n; i--, j++)
{
    if (s[i].end() != s[i].find(j))
        break;
    answer++;
}
for(int i = c_q + 1; i < n; i++)
{
    if (s[r_q].end() != s[r_q].find(i))
        break;
    answer++;
}
for(int i = c_q + 1, j = r_q + 1; i < n && j < n; i++, j++)
{
    if (s[j].end() != s[j].find(i))
        break;
    answer++;
}
for(int i = r_q + 1; i < n; i++ )
{
    if (s[i].end() != s[i].find(c_q))
        break;
    answer++;
}
for(int i = r_q + 1, j = c_q - 1; i < n && j >= 0; i++, j--)
{
    if (s[i].end() != s[i].find(j))
        break;
    answer++;
}
for(int i = c_q - 1; i >= 0; i--)
{
    if (s[r_q].end() != s[r_q].find(i))
        break;
    answer++;
}
for(int i = c_q - 1, j = r_q - 1; i >= 0 && j >= 0; i--, j--)
{
    if (s[j].end() != s[j].find(i))
        break;
    answer++;
}
return (answer);
}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    string nk_temp;
    getline(cin, nk_temp);

    vector<string> nk = split_string(nk_temp);

    int n = stoi(nk[0]);

    int k = stoi(nk[1]);

    string r_qC_q_temp;
    getline(cin, r_qC_q_temp);

    vector<string> r_qC_q = split_string(r_qC_q_temp);

    int r_q = stoi(r_qC_q[0]);

    int c_q = stoi(r_qC_q[1]);

    vector<vector<int>> obstacles(k);
    for (int i = 0; i < k; i++) {
        obstacles[i].resize(2);

        for (int j = 0; j < 2; j++) {
            cin >> obstacles[i][j];
        }

        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    int result = queensAttack(n, k, r_q, c_q, obstacles);

    fout << result << "\n";

    fout.close();

    return 0;
}

vector<string> split_string(string input_string) {
    string::iterator new_end = unique(input_string.begin(), input_string.end(), [] (const char &x, const char &y) {
        return x == y and x == ' ';
    });

    input_string.erase(new_end, input_string.end());

    while (input_string[input_string.length() - 1] == ' ') {
        input_string.pop_back();
    }

    vector<string> splits;
    char delimiter = ' ';

    size_t i = 0;
    size_t pos = input_string.find(delimiter);

    while (pos != string::npos) {
        splits.push_back(input_string.substr(i, pos - i));

        i = pos + 1;
        pos = input_string.find(delimiter, i);
    }

    splits.push_back(input_string.substr(i, min(pos, input_string.length()) - i + 1));

    return splits;
}
