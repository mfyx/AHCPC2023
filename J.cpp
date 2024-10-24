#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <numeric>
using namespace std;
using ll = long long;
const int MOD = 1e9 + 7;

// 获取数字x的质因数集合
set<int> get_prime_factors(int x) {
    if (x == 1) return set<int>();
    set<int> factors;
    int d = 2;
    while (d * d <= x) {
        while (x % d == 0) {
            factors.insert(d);
            x /= d;
        }
        d++;
    }
    if (x > 1) {
        factors.insert(x);
    }
    return factors;
}

// 识别等价类的辅助函数
vector<vector<int>> identify_equivalence_classes(int n) {
    map<set<int>, vector<int>> classes;
    for (int x = 1; x <= n; x++) {
        set<int> factors = get_prime_factors(x);
        classes[factors].push_back(x);
    }
    vector<vector<int>> result;
    for (const auto& [_, nums] : classes) {
        result.push_back(nums);
    }
    return result;
}

// 解决单个测试用例
int solve_case(int n, vector<int>& a) {
    // 识别等价类
    auto eq_classes = identify_equivalence_classes(n);
    
    // 确定已固定的槽位和可用的数
    map<int, int> fixed;
    set<int> available;
    for (int i = 1; i <= n; i++) {
        available.insert(i);
    }
    
    for (int i = 0; i < n; i++) {
        int slot = i + 1;
        if (a[i] != 0) {
            if (available.find(a[i]) == available.end()) {
                return 0;  // 重复赋值，不可能
            }
            fixed[slot] = a[i];
            available.erase(a[i]);
        }
    }
    
    // 预处理槽位之间的互质关系
    vector<set<int>> coprime_slots(n + 1);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (i != j && gcd(i, j) == 1) {
                coprime_slots[i].insert(j);
            }
        }
    }
    
    // 预处理数与数之间的互质关系
    vector<set<int>> num_coprime(n + 1);
    for (int x = 1; x <= n; x++) {
        for (int y = 1; y <= n; y++) {
            if (gcd(x, y) == 1) {
                num_coprime[x].insert(y);
            }
        }
    }
    
    // 确定需要分配的槽位和可用数
    vector<int> slots_to_assign;
    for (int s = 1; s <= n; s++) {
        if (fixed.find(s) == fixed.end()) {
            slots_to_assign.push_back(s);
        }
    }
    
    vector<int> avail_numbers(available.begin(), available.end());
    
    // 如果槽位数大于可用数，返回0
    if (slots_to_assign.size() > avail_numbers.size()) {
        return 0;
    }
    
    // 回溯法计数
    int count = 0;
    
    function<void(int, map<int, int>&, set<int>&)> backtrack = 
        [&](int index, map<int, int>& current_assignment, set<int>& used) {
            if (index == slots_to_assign.size()) {
                count = (count + 1) % MOD;
                return;
            }
            
            int slot = slots_to_assign[index];
            for (int num : avail_numbers) {
                if (used.find(num) == used.end()) {
                    // 检查与已分配的互质槽位
                    bool valid = true;
                    for (const auto& [assigned_slot, assigned_num] : current_assignment) {
                        if (coprime_slots[slot].find(assigned_slot) != coprime_slots[slot].end()) {
                            if (gcd(num, assigned_num) != 1) {
                                valid = false;
                                break;
                            }
                        }
                    }
                    
                    if (valid) {
                        // 选择num分配给slot
                        current_assignment[slot] = num;
                        used.insert(num);
                        backtrack(index + 1, current_assignment, used);
                        // 撤销选择
                        current_assignment.erase(slot);
                        used.erase(num);
                    }
                }
            }
        };
    
    // 初始化当前分配为固定分配
    map<int, int> initial_assignment = fixed;
    set<int> used_numbers;
    for (const auto& [_, val] : fixed) {
        used_numbers.insert(val);
    }
    
    backtrack(0, initial_assignment, used_numbers);
    return count;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T;
    cin >> T;
    while (T--) {
        int n;
        cin >> n;
        vector<int> a(n);
        for (int i = 0; i < n; i++) {
            cin >> a[i];
        }
        cout << solve_case(n, a) << '\n';
    }
    return 0;
}