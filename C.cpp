#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

// 定义函数，计算最小值
double calculate_min_value(double a, double b) {
    return sqrt(b * b + a + 2 * sqrt(a) + 1);
}

int main() {
    double a, b;
    cin >> a >> b;

    // 计算并输出结果，保留小数点后 6 位
    double result = calculate_min_value(a, b);
    cout << fixed << setprecision(6) << result << endl;

    return 0;
}