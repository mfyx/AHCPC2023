#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <sstream>
using namespace std;

int simulate(const vector<int>& x, const vector<int>& y, const vector<int>& t, int A, int B, char direction) {
    int step = 0;
    int current_A = A;
    int current_B = B;
    set<pair<int, int>> visited;
    
    while (visited.find({current_A, current_B}) == visited.end()) {
        if (t[current_A] != t[current_B]) {
            return step;
        }
        visited.insert({current_A, current_B});
        int next_A, next_B;
        if (direction == 'X') {
            next_A = x[current_A];
            next_B = x[current_B];
        } else {
            next_A = y[current_A];
            next_B = y[current_B];
        }
        current_A = next_A;
        current_B = next_B;
        step++;
    }
    return -1; // "GG"
}

int main() {
    int T;
    cin >> T;
    cin.ignore(); // 忽略换行符

    while (T--) {
        string line;
        getline(cin, line);
        stringstream ss(line);
        int n, A, B;
        ss >> n >> A >> B;

        vector<int> x(n), y(n), t(n);
        for (int i = 0; i < n; ++i) {
            getline(cin, line);
            stringstream ss(line);
            ss >> x[i] >> y[i] >> t[i];
        }

        int step_X = simulate(x, y, t, A, B, 'X');
        int step_Y = simulate(x, y, t, A, B, 'Y');

        if (step_X != -1 && step_Y != -1) {
            cout << min(step_X, step_Y) << endl;
        } else if (step_X != -1) {
            cout << step_X << endl;
        } else if (step_Y != -1) {
            cout << step_Y << endl;
        } else {
            cout << "GG" << endl;
        }
    }

    return 0;
}