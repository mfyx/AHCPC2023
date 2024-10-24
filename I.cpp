#include <iostream>
#include <vector>
#include <queue>
#include <iomanip>
using namespace std;

// 定义一个结构体来存储优先队列中的元素
struct HeapNode {
    double gain;
    int index;
    
    HeapNode(double g, int i) : gain(g), index(i) {}
    
    // 重载小于运算符以创建最大堆（因为我们需要最大的增益）
    bool operator<(const HeapNode& other) const {
        return gain < other.gain;
    }
};

class Solution {
private:
    // 分配物质β并计算期望值
    double allocateAndCalculate(vector<int>& p, vector<int>& l, int t, int n) {
        // 初始化分配数组
        vector<int> b(n, 0);
        
        // 构建优先队列（最大堆）
        priority_queue<HeapNode> heap;
        for (int i = 0; i < n; i++) {
            if (l[i] == 0) {
                heap.push(HeapNode(0, i));
            } else {
                double gain = static_cast<double>(p[i]) / (l[i] + 1);
                heap.push(HeapNode(gain, i));
            }
        }
        
        // 分配t单位的物质β
        for (int i = 0; i < t && !heap.empty(); i++) {
            HeapNode top = heap.top();
            heap.pop();
            int idx = top.index;
            
            // 为容器i分配一个单位
            b[idx]++;
            
            if (b[idx] < l[idx]) {
                // 计算新的边际增益
                double new_gain = static_cast<double>(p[idx] * l[idx]) / 
                                ((l[idx] + b[idx] + 1) * (l[idx] + b[idx] + 1));
                heap.push(HeapNode(new_gain, idx));
            }
        }
        
        // 计算总期望值
        double total_E = 0.0;
        for (int i = 0; i < n; i++) {
            if (b[i] > 0) {
                total_E += static_cast<double>(b[i]) / (l[i] + b[i]) * p[i];
            }
        }
        
        return total_E;
    }

public:
    void solve() {
        int n, t, q;
        cin >> n >> t >> q;
        
        // 读入数据
        vector<int> p(n);
        vector<int> l(n);
        
        for (int i = 0; i < n; i++) {
            cin >> p[i];
        }
        for (int i = 0; i < n; i++) {
            cin >> l[i];
        }
        
        // 处理查询
        for (int i = 0; i < q; i++) {
            int tj, rj;
            cin >> tj >> rj;
            rj--; // 转换为0基索引
            
            // 更新l[rj]
            if (tj == 1) {
                l[rj]++;
            } else {
                l[rj]--;
            }
            
            // 计算并输出结果
            double result = allocateAndCalculate(p, l, t, n);
            cout << fixed << setprecision(9) << result << endl;
        }
    }
};

int main() {
    // 设置IO优化
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    Solution solution;
    solution.solve();
    
    return 0;
}