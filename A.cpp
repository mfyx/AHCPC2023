#include <iostream>
using namespace std;

int years_until_pollution(int N, int M, int K) {
    int years = 0;
    double current_area = N;
    while (current_area > M) {
        current_area *= (1 - K / 100.0);
        years++;
    }
    return years;
}

int main() {
    int N, M, K;
    cin >> N >> M >> K;

    int result = years_until_pollution(N, M, K);

    cout << result << endl;

    return 0;
}
