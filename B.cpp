#include <iostream>
using namespace std;

void solve() {
    int T;
    cin >> T;
    while (T--) {
        long long x;
        cin >> x;
        long long a = x / 3;
        long long b = x % 3;
        if (x == 1) {
            cout << 3 << endl;
            continue;
        }
        if (b == 1) {
            cout << (a + 1) * 4 << endl;
        } else if (b == 2) {
            cout << (a + 1) * 4 + 1 << endl;
        } else if (b == 0) {
            cout << (a + 1) * 4 - 1 << endl;
        }
    }
}

int main() {
    solve();
    return 0;
}