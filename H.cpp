#include <iostream>
#include <vector>
#include <climits>
#include <cstring>
using namespace std;

class MaxFlow {
public:
    MaxFlow(vector<vector<int>>& graph) : graph(graph), ROW(graph.size()) {}

    bool dfs(int s, int t, vector<int>& parent) {
        vector<bool> visited(ROW, false);
        vector<int> stack;
        stack.push_back(s);
        visited[s] = true;

        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();

            for (int v = 0; v < ROW; ++v) {
                if (!visited[v] && graph[u][v] > 0) {
                    stack.push_back(v);
                    visited[v] = true;
                    parent[v] = u;

                    if (v == t) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    int ford_fulkerson(int source, int sink) {
        vector<int> parent(ROW, -1);
        int max_flow = 0;

        while (dfs(source, sink, parent)) {
            int path_flow = INT_MAX;
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                path_flow = min(path_flow, graph[u][v]);
            }

            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                graph[u][v] -= path_flow;
                graph[v][u] += path_flow;
            }

            max_flow += path_flow;
        }
        return max_flow;
    }

private:
    vector<vector<int>>& graph;
    int ROW;
};

int main() {
    int N, M;
    cin >> N >> M;
    vector<vector<int>> graph(N, vector<int>(N, 0));

    for (int i = 0; i < M; ++i) {
        int u, v, capacity;
        cin >> u >> v >> capacity;
        graph[u - 1][v - 1] = capacity;  // 将站点编号调整为 0 基数
    }

    MaxFlow max_flow_solver(graph);
    int source = 0;  // 源点为发电站（编号1，索引0）
    int sink = N - 1;  // 汇点为变电站（编号N，索引N-1）

    int result = max_flow_solver.ford_fulkerson(source, sink);
    cout << result << endl;

    return 0;
}