#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
private:
    vector<vector<int>> graph;
    vector<bool> visited;
    vector<long long> sub;
    vector<long long> f;
    vector<long long> g;
    vector<int> values;
    
    void dfs(int node) {
        // 如果节点无效，直接返回
        if (node == 0) return;
        
        // 检查是否还有未访问的相邻节点
        bool flag = false;
        for (int i : graph[node]) {
            if (!visited[i]) {
                flag = true;
                break;
            }
        }
        if (!flag) return;
        
        // 标记当前节点为已访问
        visited[node] = true;
        
        // 递归访问所有未访问的相邻节点
        for (int i : graph[node]) {
            if (!visited[i]) {
                dfs(i);
            }
        }
        
        // 计算g[node]
        g[node] = 0;
        for (int i : graph[node]) {
            g[node] += sub[i];
        }
        
        // 计算f[node]
        long long maxf = 0;
        for (int i : graph[node]) {
            long long ft = g[i] + (long long)values[i] * values[node];
            for (int j : graph[node]) {
                if (j != i) {
                    ft += sub[j];
                }
            }
            maxf = max(ft, maxf);
        }
        f[node] = maxf;
        
        // 更新sub[node]
        sub[node] = max(f[node], g[node]);
    }

public:
    long long solve(int n, vector<pair<int, int>>& roads, vector<int>& input_values) {
        // 初始化所有数组
        graph.resize(n + 1);
        visited.resize(n + 1, false);
        sub.resize(n + 1, 0);
        f.resize(n + 1, 0);
        g.resize(n + 1, 0);
        values = vector<int>(n + 1, 0);
        
        // 复制输入的values数组，注意从下标1开始
        for (int i = 1; i <= n; i++) {
            values[i] = input_values[i-1];
        }
        
        // 构建图
        for (const auto& road : roads) {
            graph[road.first].push_back(road.second);
            graph[road.second].push_back(road.first);
        }
        
        // 从根节点1开始DFS
        dfs(1);
        
        return sub[1];
    }
};

int main() {
    int n;
    cin >> n;
    
    // 读入道路信息
    vector<pair<int, int>> roads;
    for (int i = 0; i < n-1; i++) {
        int a, b;
        cin >> a >> b;
        roads.push_back({a, b});
    }
    
    // 读入节点值
    vector<int> values(n);
    for (int i = 0; i < n; i++) {
        cin >> values[i];
    }
    
    Solution solution;
    cout << solution.solve(n, roads, values) << endl;
    
    return 0;
}
