#include <iostream>
#include <vector>
#include <string>
#include <sstream>
using namespace std;

int main() {
    string first_line;
    getline(cin, first_line);
    stringstream ss(first_line);
    int n, k;
    ss >> n >> k;

    vector<vector<int>> a(n + 1, vector<int>(n + 1, 0));
    vector<int> h(n + 1, 0);
    vector<int> v(n + 2, 0);

    for (int s = 1; s < n; ++s) {
        string line;
        getline(cin, line);
        stringstream ss(line);
        for (int t = s + 1; t <= n; ++t) {
            ss >> a[s][t];
            h[s] += a[s][t];
        }
    }

    for (int t = 2; t <= n; ++t) {
        for (int s = 1; s < n; ++s) {
            v[t] += a[s][t];
        }
    }

    int window = 0;
    for (int i = 1; i <= k; ++i) {
        window += h[i];
    }
    // cout << "window = " << window << endl;

    int ans = window;

    for (int i = 2; i <= n - k; ++i) {
        window -= v[i];
        window += h[i + k - 1];
        // cout << "window = " << window << endl;
        ans = max(window, ans);
    }

    cout << ans << endl;

    return 0;
}