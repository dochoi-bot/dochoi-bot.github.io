#include <bits/stdc++.h>

using namespace std;

vector<string> split_string(string);

// Complete the climbingLeaderboard function below.
vector<int> climbingLeaderboard(vector<int> scores, vector<int> alice) {

  vector<int> answer;


    for(int i = 1; i < scores.size();)
    {
        if (scores[i] == scores[i - 1])
            scores.erase(scores.begin() + i);
        else
            i++;
    }
      int base = 1;
int sc_size=  scores.size();
while (base <= sc_size - 1)
    base *= 2;
    for(int i = 0; i < alice.size();  i++)
    {  
        int idx = base - 1;
        if (alice[i] >= scores[0])
        {
             answer.push_back(1);
            continue;
        }
        else if (alice[i] < scores[sc_size - 1])
        {
             answer.push_back(sc_size + 1);
            continue;
        }
        else if (alice[i] == scores[sc_size - 1])
        { answer.push_back(sc_size);
            continue;
        }
        int add = base / 2;
        while(true)
        {
            if (idx < 0)
                idx += add;
            else if (idx >= sc_size)
                idx -= add;
            else if (alice[i] > scores[idx])
                idx -= add;
            else if (alice[i] < scores[idx])
                idx += add;
            else if (alice[i] == scores[idx])
            {
                break;
            }
            if (add == 0)
                break; 
                add /= 2;
        }
        if (scores[idx] > alice[i])
        {
        answer.push_back(idx + 2);
        }
        else if (scores[idx] <= alice[i])
        {
            answer.push_back(idx + 1);
        }
    }
 
return(answer);
}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    int scores_count;
    cin >> scores_count;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    string scores_temp_temp;
    getline(cin, scores_temp_temp);

    vector<string> scores_temp = split_string(scores_temp_temp);

    vector<int> scores(scores_count);

    for (int i = 0; i < scores_count; i++) {
        int scores_item = stoi(scores_temp[i]);

        scores[i] = scores_item;
    }

    int alice_count;
    cin >> alice_count;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    string alice_temp_temp;
    getline(cin, alice_temp_temp);

    vector<string> alice_temp = split_string(alice_temp_temp);

    vector<int> alice(alice_count);

    for (int i = 0; i < alice_count; i++) {
        int alice_item = stoi(alice_temp[i]);

        alice[i] = alice_item;
    }

    vector<int> result = climbingLeaderboard(scores, alice);

    for (int i = 0; i < result.size(); i++) {
        fout << result[i];

        if (i != result.size() - 1) {
            fout << "\n";
        }
    }

    fout << "\n";

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
