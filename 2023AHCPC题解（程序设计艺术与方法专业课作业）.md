# 2023AHCPC题解（程序设计艺术与方法专业课小组作业）

> 本文为hfut程序设计艺术与方法专业课小组作业题目的题解
> 题目可参考该[OJ](https://acm.webturing.com/contest.php?cid=1809)

## **写在前面**

*关于OJ*
本来以为题目可通过该[OJ](https://acm.webturing.com/contest.php?cid=1809)提交, 后来发现这个OJ貌似没有新增任何测试数据, 即使仅仅输出样例数据的输出也可AC, 故该OJ几乎不具有使用价值.
目前未在网上发现提供完整测试数据的OJ, 这也意味着**没法完全保证本题解是正确的**, 请批判看待本题解.

*关于D题*
本题解之前未给出D题解答, 现已给出.

*关于F题和G题*
**本题解之前给出的F题和G题的思路和代码都是错误的! 现已更新, 目前应该是对的了**

*关于I题*
尽管基于题解中的思路, I题可以得到正确结果, 但是把题目给定的数据范围拉满, 会超时, 暂未找到不超时的思路.

*关于K题*
K题题目是没问题的, 但提供的输入样例不符合体面要求的格式, 本题解没有更新K题的打算.

## A 植被保护

**题目复述**

第一年H市的植被覆盖数为N平方千米，当植被覆盖数达到M平方千米以下时，则说明当地已经严重污染， 植被覆盖不足。假设H市的植被覆盖数每年以K%的减少。请问多少年之后H市将会严重污染?

**测试样例**

*输入*
一行包含三个整数  N,M,K(1≤N,M≤2³¹−1,1≤K<100),N表示第一年H市的植被覆盖数，M表示M平方千米以下时，则说明当地已经严重污染， K 表示H市的植被覆盖数每年以K%的减少

*输出*
输出一个整数， 表示多少年之后H市将会严重污染。

*样例输入*
```
19 10 50
```

*样例输出*
```
1
```

**样例解析**

$$
19*(1-\frac{50}{100}) = 9.5 < 10
$$
故一年之后植被覆盖就不达标了，进入严重污染状态。

**参考代码**

``` python
def years_until_pollution(N, M, K):
    years = 0
    current_area = N
    while current_area > M:
        current_area *= (1 - K / 100)
        years += 1
    return years

N, M, K = map(int, input().split())

result = years_until_pollution(N, M, K)

print(result)
```

---

## B 环保数列

**题目复述**

环保数n是指能够写成$a^2-b^2=n$的数（a, b均为正整数），
从小到大所有环保数依次排列组成环保数列，
请求出环保数列第x项的值。

**测试样例**

*输入*
第一行一个整数 T(1<=T<=10), 表示数据组数;对于每组数据,输入一行一个整数x (1<=x<=10^9),表示询问的项数.

*输出*
对于每组数据,输出一行一个整数,表述题述定义的神奇数列第x项的值.

*样例输入*
```
2
4
6
```

*样例输出*
```
8
11
```

**样例解析**

从小到大前10项环保数依次为`3 5 7 8 9 11 12 13 15 16`, 故第4项为8, 第6项为11.

**思路解析**

$a^2-b^2=(a+b)(a-b)$, 可分为如下三种情况:
- a和b都为奇数, 则a+b和a-b都为偶数
- a和b都为偶数, 则a+b和a-b都为偶数
- a和b一奇一偶, 则a+b和a-b都为奇数
故当且仅当n可以写成两个不同的偶数相乘或两个不同的奇数相乘时, n为环保数(注意必须是不同的数, 因为`b!=0`)
对于把n写成两个偶数相乘的情况, 两个因数各提因子2, 可知, n为4的倍数.
综上, 当n为奇数或者n为4的倍数时, n为环保数(4除外, 因为4不能写成不同的偶数相乘)
基于该规律, 可找到如下通项公式: 
$$
f(x) =
\begin{cases}
3, & \text{if } x = 1 \\
(a + 1) \times 4, & \text{if } x \equiv 1 \pmod{3} \\
(a + 1) \times 4 + 1, & \text{if } x \equiv 2 \pmod{3} \\
(a + 1) \times 4 - 1, & \text{if } x \equiv 0 \pmod{3}
\end{cases}
$$

**参考代码**

``` python
def main():
    T = int(input())
    for _ in range(T):
        x = int(input().strip())
        a = x // 3
        b = x % 3
        if x == 1:
            print(3)
            continue
        if b == 1:
            print((a+1)*4)
        if b == 2:
            print((a+1)*4+1)
        if b == 0:
            print((a+1)*4-1)
        
main()
```

---

## C 联合密码

**题目复述**

给定a和b, 求 $\sqrt{x^2+a}$ 和 $\sqrt{(b-x)^2+1}$ 的最小值, x可以为任意实数.

**测试样例**

*输入*
输入a, b两个非负实数

*输出*
输出表达式的最小值, 精确到小数点后6位.

*样例输入*
```
4.000 4.000
```

*样例输出*
```
5.000000
```

**思路解析**

将x对应到二维坐标平面中的点(x, 0), 则$\sqrt{x^2+a}$即为(x, 0)到(0, √a)的距离, 而$\sqrt{(b-x)^2+1}$即为(x, 0)到(b, 1)的距离.
取(b, 1)关于x轴的对称点(b, -1), (0, √a)与(b, -1)所连成的线段即为次最小距离.
故答案为$\sqrt{b^2+(\sqrt{a}+1)^2}$
![示意图](https://i-blog.csdnimg.cn/blog_migrate/70d41bb3647301b6d759ccab6f5e90fe.png)
([图片来源](https://blog.csdn.net/qq_74363674/article/details/131614338))

**样例解析**

(0, 2)到(4, -1)的距离为5, 故输出5

**参考代码**

``` python
from math import sqrt

# 定义函数，计算最小值
def calculate_min_value(a, b):
    ret = sqrt(b**2 + a + 2*sqrt(a) + 1)
    return ret

# 读取输入
a, b = map(float, input().split())

# 计算并输出结果，保留小数点后 6 位
result = calculate_min_value(a, b)
print(f"{result:.6f}")
```

---

## D 环境宣传

**题目复述**

政府选择了一条道路将作为一个宣传试验场地，以促进可持续交通，减少该市的碳足迹。这条道路可以看成一条直线，上面有 N 个和其他道路交错形成的路口，每个相邻的路口之间可以安装装置宣传到经过此段道路的市民。由于装置的价值昂贵，所以不能在每个相邻的路口之间安装装置进行宣传，所以政府决定选定 k 个相邻的路口，在路口之间安装装置进行宣传。关于每天在每对路口之间通行的市民数量的统计数据已经知晓(假设每位市民每天只通行一次，且从一个路口进，一个路口出)。现在，政府需要知道在哪些路口之间安装装置可以使得最多市民受到宣传，促进可持续交通和减少碳排放。请你帮忙计算收到宣传的最大市民数是多少。

**测试样例**

*输入*
第一行包含两个整数 n , k ，表示道路经过的路口数和可以安装的装置数目。
接下来 n −1 行，每行包含 n i − (1<=i<=n-1)个整数，其中第 j
（1<=j<=n-i） 
个数表示第 i 个路口到第 i j + 个路口之间的每天的通行市民数量。

*输出*
第一行包含一个整数，表示能够收到宣传的市民最大总数。

*样例输入*
```
4 1
5 0 6
5 3
5
```

*样例输出*
```
14
```

*样例解析*
在1, 2路口之间安装装置, 有 5+0+6=11 个市民受到宣传
在2, 3路口之间和3, 4路口之间安装装置，都有 6+3+5=14 个市民受到宣传. 
故最多有14个市民受到宣传

*自造测试输入*
```
4 2
5 0 6
5 3
5
```

*自造测试输出*
```
19
```

*样例解析*
一共有两种情况:
情况①: 在1, 2和2, 3路口之间安排装置, 有 19 个市民受到宣传.
情况②: 在2, 3和3, 4路口之间安排装置, 有 19 个市民受到宣传.

**思路解析**

找规律发现, 可以通过滑动窗口来解决这个问题, 
但本题滑动窗口滑动的方式并非传统的从左侧缩短, 从右侧伸长,
而是将数据表示成二维数组, 通过减去数组中左侧的列的和来缩短, 通过加上数组中下侧的行的和来伸长
其中, k为滑动窗口的大小, 初始时滑动窗口的值为数组中前k行的和
以两份测试数据为例, 首先将输入数据右对齐:
```
5 0 6
  5 3
    5
```

对于k=1的情况, 
初始时滑动窗口为第一行的和: 5+0+6=11, 代表着在1, 2路口之间安装装置时受到宣传的市民人数.
然后去掉第一列的数, 算上第二行的数: 0+6+5+3=14, 代表着在2, 3路口之间安装装置时受到宣传的市民人数.
然后去掉第二列的数, 算上第三行的数: 6+3+5=14, 代表着在3, 4路口之间安装装置时受到宣传的市民人数.
(始终去掉剩余列中最左侧的列, 加上最下方的行的下一行)

对于k=2的情况,
初始时滑动窗口为第一行和第二行的和: 5+0+6+5+3 = 19, 代表着在1, 2路口之间和2, 3路口之间安装装置时受到宣传的市民人数.
然后去掉第一列的数, 算上第三行的数: 0+6+5+3+5 = 19, 代表着在2, 3路口之间和3, 4路口之间安装装置时受到宣传的市民人数.

> 之前看到一篇[csdn题解](https://blog.csdn.net/qq_74363674/article/details/131614338), 测试发现不能通过本题的自造样例, 故该题解的D题应该是错的

**参考代码**

``` python
def main():
    first_line = input().strip().split()
    n = int(first_line[0])
    k = int(first_line[1])

    a = [[0] * (n + 1) for _ in range(n + 1)]
    h = [0] * (n + 1)
    v = [0] * (n + 2)

    for s in range(1, n):
        line = input().strip().split()
        for t in range(s+1, n+1):
            a[s][t] = int(line[t - (s+1)])
            h[s] += a[s][t]
    
    for t in range(2, n+1):
        for s in range(1, n):
            v[t] += a[s][t]
    
    window = 0
    for i in range(1, 1+k):
        window += h[i]
    print("window = " + str(window))
    
    ans = window

    for i in range(2, n-k+1):
        window -= v[i]
        window += h[i + k - 1]
        print("window = " + str(window))
        ans = max(window, ans)
    
    print(ans)

main()
```

---

## E 城市规划

**题目复述**
在某市有 n 个路口，每个路口都连接着另外两个路口，可以向方向 X 行走到
达某个路口，或向方向 Y 行走到达某个路口（可能相同也可能回到原地），所有的路口被分为两种类型（用 0/1 表示），路口编号为 0 到 n-1。现在以“路口独特度”指标评价该市的城市规划合理性。从 A 和 B 两个路口出发，一直按照同样的方向模拟从 A 路口出发和从 B 路口出发走，直到走到种类不同的路口，所需要的最短步数就是“路口独特度”。现在，给出该市的地图，请求出“路口独特度”。

**测试样例**

*输入*
输入包含多组数据，第一行输入数据组数 T。每组数据的输入如下：
第一行三个正整数：n,A,B（A≠B）
第二行到第 n+1 行每行三个整数：xi, yi, ti，表示路口 i 向方向 X 走到达路口 xi，
向方向 Y 走到达路口 yi，它的种类为 ti。

*输出*
如果能够判断，则输出最少步数，否则输出 GG

*样例输入*
```
2 
3 1 2 
1 2 1 
0 2 0 
0 1 0 
3 1 2 
1 2 0 
2 0 1 
0 1 1
```

*样例输出*
```
GG
1
```

**样例解析**

*测试用例1*

| 路口编号 | X方向连接 | Y方向连接 | 类型 (`ti`) |
|----------|----------|----------|-------------|
| 0        | 1        | 2        | 1           |
| 1        | 0        | 2        | 0           |
| 2        | 0        | 1        | 0           |

分别从路口1和路口2出发, 如果沿X走, 则走1步后二者都走到路口0, 故不可能类型不同; 如果沿Y走, 二者位置互换, 继续不断走, 位置将不断互换, 故不可能类型不同.
综上, 输出GG.

*测试用例2*

| 路口编号 | X方向连接 | Y方向连接 | 类型 (`ti`) |
|----------|----------|----------|-------------|
| 0        | 1        | 2        | 0           |
| 1        | 2        | 0        | 1           |
| 2        | 0        | 1        | 1           |

分别从路口1和路口2出发, 如果沿X走, 则走1步后分别到路口2和路口0, 二者类型不同; 如果沿Y走, 则走1步后分别到路口0和路口1, 二者类型不同.
综上, 遇到类型不同的情况所需的最小步数是1, 故输出1.

**题意概括**

编号为0到n-1的n个路口之间, 通过X方向和Y方向的道路进行连接, 
每个路口的类型为0或1
现在指定两个不同的路口A和B, 从A和B分别出发, 一直沿着某个方向一直走(一直沿X或者一直沿Y), 求最少多少步时, 可以走到类型不同的路口
如果走不到类型不同的路口, 输出GG, 否则输出最小步数

**思路解析**

分别模拟从A, B为起点沿X方向和Y方向一直走, 记录行进过程中遇到过的路口对, 当遇到相同路口对时, 说明进入循环, 结果为GG; 当遇到不同类型的路口时, 记录步数并结束循环.
记录两种方案中更优的方案, 并输出.

**参考代码**
``` python
def main():
    T = int(input())
    results = []

    for _ in range(T):
        # 读取 n, A, B
        while True:
            line = input().strip()
            if line == '':
                continue  # 跳过空行
            else:
                break
        n_A_B = line.split()
        while len(n_A_B) < 3:
            n_A_B += input().strip().split()
        n, A, B = map(int, n_A_B)

        # 读取每个路口的信息
        x = [0] * n
        y = [0] * n
        t = [0] * n
        count = 0
        while count < n:
            line = input().strip()
            if line == '':
                continue  # 跳过空行
            parts = line.split()
            while len(parts) < 3:
                parts += input().strip().split()
            xi, yi, ti = map(int, parts)
            x[count] = xi
            y[count] = yi
            t[count] = ti
            count += 1

        def simulate(direction):
            step = 0
            current_A = A
            current_B = B
            visited = set()
            while (current_A, current_B) not in visited:
                if t[current_A] != t[current_B]:
                    return step
                visited.add((current_A, current_B))
                if direction == 'X':
                    next_A = x[current_A]
                    next_B = x[current_B]
                else:
                    next_A = y[current_A]
                    next_B = y[current_B]
                current_A, current_B = next_A, next_B
                step += 1
            return "GG"

        step_X = simulate('X')
        step_Y = simulate('Y')

        # 确定最小步数
        min_step = None
        if step_X != "GG" and step_Y != "GG":
            min_step = min(step_X, step_Y)
        elif step_X != "GG":
            min_step = step_X
        elif step_Y != "GG":
            min_step = step_Y

        if min_step is not None:
            results.append(str(min_step))
        else:
            results.append("GG")

    # 输出所有结果
    for res in results:
        print(res)

main()
```

---

## F 太阳能板

**题目复述**

科研人员想要研发新的太阳能板材料，在 n 个仓库中存放了 n 种原始
材料，有 n-1 条道路将这 n 个仓库连接在一起，每条道路连接着两个仓库，进行
新材料的合成必须使用两种原始材料。考虑到运输成本和材料成本，只能选择被
一条道路直接相连的两个仓库中的原始材料进行合成，且每种原始材料只能被使
用一次。在两种原始材料合成之后，得到的新材料的吸光能力为两种原始材料的
吸光能力之乘积。现在科研人员想要知道，合成的新材料的吸光能力的总和最大是多少。

**测试样例**

*输入*
第一行一个正整数：n 
第二行到第 n 行每行两个整数：ai, bi 表示仓库 ai 和仓库 bi 之间存在道路，
第 n+1 行 n 个正整数：vi，表示仓库 i 中的原始材料的吸光能力

*输出*
能够得到的最大的吸光能力总和。

*样例输入*
```
5 
1 2
1 3 
2 4 
2 5 
1 2 3 4 5
```

*样例输出*
```
13
```

**样例解析**

道路连接构成的树如下：
```
   1
  / \
 2   3
/ \
4  5
```
首先选择节点2和节点5进行合并, 得到合成材料的吸光能力$2*5=10$
然后选择节点1和几点3进行合并, 得到合成材料的吸光能力$1*3=3$
现在我们已经使用了所有可以使用的边，并且每个材料只使用了一次。
故总的合成材料吸光能力为$10+3=13$

**题意概括**

给一个树的每个节点i分配一个权值 g[vi], 可以合并树中两个节点, 得到 g[v1]*g[v2] 的吸光能力, 每个节点只能合并一次, 求最多可以得到多少吸光能力.

**思路解析**

采用树形动态规划, 维护 f, g, sub 三个DP数组, 
f[i]表示以i为根节点的子树在选择节点i与其子节点进行合并的情况下所能得到的最大吸光能力,
g[i]表示以i为根节点的子树在不选择节点i与其子节点进行合并的情况下所能得到的最大吸光能力,
sub[i]表示以i为根节点的子树所能得到的最大吸光能力.

可得三个DP数组的状态转移方程如下(假设节点i的子节点为 $j_1, j_2, ..., j_n$, f转移方程最后的sub求和代表不被合并的各个子节点所对应的子树的总吸光能力):
$$
f[i] = max{ g[j_1] + i*j_1 + (sub(j_2) + sub(j_3) + ... + sub(j_n)),
            g[j_2] + i*j_2 + (sub(j_1) + sub(j_3) + ... + sub(j_n)),
            ...
            g[j_n] + i*j_n + (sub(j_1) + sub(j_2) + ... + sub(j_{n-1}))
          }
g[i] = sub[j_1] + sub[j_2] + ... + sub[j_n]
sub[i] = max{f[i], g[i]}
$$

采用深搜更新状态转移方程, 最后得到的sub[start]即为所求

> 之前尝试用贪心策略解决本题, 后来发现贪心策略无法解决本问题, 应使用DP, 故之前给出的思路和代码都是错的

**参考代码**

``` python
from collections import defaultdict

def main():
    """
    读入数据, 并初始化变量
    """
    n = int(input().strip())
    roads = []

    for _ in range(n - 1):
        a, b = map(int, input().strip().split())
        roads.append((a, b))
    values = list(map(int, input().strip().split()))
    values.insert(0, 0)  # 补0占位

    graph = defaultdict(list)
    for a, b in roads:
        graph[a].append(b)
        graph[b].append(a)

    visited = [False] * (n + 1)
    sub = [0] * (n + 1)
    f = [0] * (n + 1)
    g = [0] * (n + 1)

    """
    定义dfs函数
    """
    def dfs(node):
        if not node:
            return 
        
        flag = False
        for i in graph[node]:
            if not visited[i]:
                flag = True
        if not flag:
            return
        
        visited[node] = True

        for i in graph[node]:
            if not visited[i]:
                dfs(i)
        
        for i in graph[node]:
            g[node] += sub[i]

        maxf = 0
        for i in graph[node]:
            ft = g[i] + values[i] * values[node]
            for j in graph[node]:
                if j != i:
                    ft += sub[j]
            maxf = max(ft, maxf)
        f[node] = maxf
        
        sub[node] = max(f[node], g[node])

    """
    调用dfs进行求解
    """
    dfs(1)
    print(sub[1])

main()
```

---

## G 动物保护

**题目复述**

今天无人机需要进行空中巡逻，需要找到无人机起
飞 A 点和投送位置 B 点之间最短距离，同时避开保护装置的影响范围，保护装
置可以看作为一个圆心为 C 点，小于半径为 R 的实心球，A 点和 B 点不会在球内。

**测试样例**

*输入*
前三行每行三个整数，表示 A、B 和 C 点的坐标。保证坐标点都在（-1000，
1000）
第四行一个整数表示装置的半径 R。

*输出*
输出 A 点到 B 的最小长度，精确到小数点后 2 位。

*样例输入*
```
0 0 12 
12 0 0 
10 0 10 
10
```

*样例输出*
```
19.71
```

**题意概括**

求三维坐标面中A点和B点不穿过球C的情况下的最短连线.

**思路解析**

对于A, B不贯穿球C的情况, 直接求线段AB的长度即可
对于A, B贯穿球C的情况,
最终的最短连线一定在A, B, C三点所确定的平面上,
故把三维坐标转换到A, B, C三点所在的平面的二维坐标,
然后在二维坐标系下求解绕行路线长度即可.

> 本文之前给出的思路是统一投影到xOz坐标平面上, 然后在xOz坐标平面求解绕行长度
> 但最短绕行路线完全可能不在xOz平面, 所以该思路是错的
> 由于本题测试样例的A, B, C三点恰好在xOz平面上, 故该思路可以得到正确的样例输出
> 正确的思路应该是投影到A, B, C三点所构成的平面, 而非xOz平面

**参考代码**

``` python
import math

def read_point():
    return list(map(int, input().strip().split()))

def vector_subtract(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])

def vector_cross(v1, v2):
    return (
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    )

def vector_dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def vector_magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def vector_normalize(v):
    mag = vector_magnitude(v)
    return (v[0] / mag, v[1] / mag, v[2] / mag)

def project_to_2d(A, B, C):
    # Calculate vectors AB and AC
    AB = vector_subtract(B, A)
    AC = vector_subtract(C, A)
    
    # Calculate normal vector to the plane
    normal = vector_cross(AB, AC)
    
    # Normalize AB to use as the first basis vector
    u = vector_normalize(AB)
    
    # Calculate the second basis vector v, orthogonal to both u and the normal
    v = vector_cross(normal, u)
    v = vector_normalize(v)
    
    # Project points onto the 2D plane defined by u and v
    def project_point(P):
        AP = vector_subtract(P, A)
        x = vector_dot(AP, u)
        y = vector_dot(AP, v)
        return (x, y)
    
    A_2d = project_point(A)
    B_2d = project_point(B)
    C_2d = project_point(C)
    
    return A_2d, B_2d, C_2d

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def distance_point_to_line(A, B, C):
    # A, B, C are tuples (x, y)
    cross = abs((B[0] - A[0]) * (A[1] - C[1]) - (B[1] - A[1]) * (A[0] - C[0]))
    dist = cross / distance(A, B)
    return dist

def tangent_points(P, C, R):
    # P and C are tuples (x, y)
    dx = P[0] - C[0]
    dy = P[1] - C[1]
    dist = math.hypot(dx, dy)
    if dist < R:
        return []  # No tangent
    elif dist == R:
        return [P]  # One tangent point (the point itself)
    else:
        angle_PC = math.atan2(dy, dx)
        alpha = math.acos(R / dist)
        t1 = angle_PC + alpha
        t2 = angle_PC - alpha
        tp1 = (C[0] + R * math.cos(t1), C[1] + R * math.sin(t1))
        tp2 = (C[0] + R * math.cos(t2), C[1] + R * math.sin(t2))
        return [tp1, tp2]

def angle_between(C, P1, P2):
    # C is center, P1 and P2 are points on circle
    v1x = P1[0] - C[0]
    v1y = P1[1] - C[1]
    v2x = P2[0] - C[0]
    v2y = P2[1] - C[1]
    dot = v1x * v2x + v1y * v2y
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 == 0 or mag2 == 0:
        return 0
    cos_theta = dot / (mag1 * mag2)
    # Clamp due to floating point
    cos_theta = max(min(cos_theta, 1), -1)
    theta = math.acos(cos_theta)
    return theta

def compute_path(A, B, C, R):
    # A, B, C are tuples (x, y)
    dist_AB = distance(A, B)
    dist_to_line = distance_point_to_line(A, B, C)
    if dist_to_line >= R:
        return dist_AB
    # Compute tangent points
    tangents_A = tangent_points(A, C, R)
    tangents_B = tangent_points(B, C, R)
    if not tangents_A or not tangents_B:
        # No possible path
        return None
    min_path = float('inf')
    for ta in tangents_A:
        for tb in tangents_B:
            # Compute angles for arc
            angle = angle_between(C, ta, tb)
            # Two possible arcs, choose the smaller one
            arc = min(angle, 2 * math.pi - angle) * R
            path = distance(A, ta) + arc + distance(B, tb)
            if path < min_path:
                min_path = path
    return min_path

def main():
    # Read input
    A = tuple(read_point())
    B = tuple(read_point())
    C = tuple(read_point())
    R = int(input())
    
    # Project points to 2D plane
    A_2d, B_2d, C_2d = project_to_2d(A, B, C)
    
    # Compute the shortest path in the 2D plane
    path_length = compute_path(A_2d, B_2d, C_2d, R)
    if path_length is None:
        print("No valid path")
    else:
        print("{0:.2f}".format(path_length))

main()
```

---

## H 电力输送

**题目复述**

为了满足东部城市对清洁能源
的需求，中国西电公司计划建设一套输电系统，从西部的可再生能源发电站将电
力输送到东部城市。该系统包括发电站、输电站和变电站等站点和一套输电线路，
用于将电力从一个地方输送到另一个地方。每个站点都被编号：发电站的编号为
1，变电站的编号为 N，中间的输电站的编号从 2 到 N-1。沿途的每段输电线路连
接一对输电站，每条输电线路可以向任意方向输送有限数量的电力。
你需要根据地图和输电线路容量，计算该输电系统最多可以输送多少电力。

**测试样例**

*输入*
第一行包含两个个整数 , N M ，其中 N 表示发电站、输电站和变电站等站点
数量之和，M 表示站点间的输电线路数量。2 10000 N≤ ≤ 。接下来的 M 行描述
了输电线路的规格：对于每条输电线路，三个数字描述连接它的一对站点以及其
输电容量 1 到108
的整数。保证输送的电量不会是负数。

*输出*
只有一行，在第一行输出可以输送的最大电力数量。

*样例输入*
```
3 2
1 2 2 
2 3 1
```

*样例输出*
```
1
```

**样例解析**

连接情况如图所示, 显然, 支持的最大流量为1
```
①---2--->②---1--->③
```

**题意概括**

最大流问题, 求一个图中起点s到终点t的最大流.

**思路解析**

用 Ford-Fulkerson 算法求解最大流即可, 详见:
[Ford-Fulkerson Algorithm 寻找网络最大流](https://www.bilibili.com/video/BV1Pv41157xh/?spm_id_from=333.880.my_history.page.click&vd_source=5b4c141029b3d309804a79b56a218572)

**参考代码**

``` python
# Ford-Fulkerson Algorithm using DFS to solve the maximum flow problem
class MaxFlow:
    def __init__(self, graph):
        self.graph = graph  # original graph with capacities
        self.ROW = len(graph)

    # A DFS based function to find if there is a path from source 's' to sink 't'
    # in residual graph. Also fills parent[] to store the path
    def dfs(self, s, t, parent):
        visited = [False] * self.ROW
        stack = [s]
        visited[s] = True

        while stack:
            u = stack.pop()

            for v, capacity in enumerate(self.graph[u]):
                if visited[v] == False and capacity > 0:
                    stack.append(v)
                    visited[v] = True
                    parent[v] = u

                    if v == t:
                        return True

        return False

    # Returns the maximum flow from s to t in the given graph
    def ford_fulkerson(self, source, sink):
        parent = [-1] * self.ROW  # Array to store the path
        max_flow = 0  # There is no flow initially

        # Augment the flow while there is a path from source to sink
        while self.dfs(source, sink, parent):

            # Find the maximum flow through the path found by DFS
            path_flow = float('Inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.graph[u][v])
                v = parent[v]

            # Update residual capacities of the edges and reverse edges along the path
            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

            # Add the path flow to overall flow
            max_flow += path_flow

        return max_flow

# Main function to run the solution
def main():
    # 输入解析
    N, M = map(int, input().split())
    # 创建容量图
    graph = [[0] * N for _ in range(N)]

    # 填充图的边信息
    for _ in range(M):
        u, v, capacity = map(int, input().split())
        graph[u - 1][v - 1] = capacity  # 将站点编号调整为 0 基数

    # 创建最大流实例
    max_flow_solver = MaxFlow(graph)
    source = 0  # 源点为发电站（编号1，索引0）
    sink = N - 1  # 汇点为变电站（编号N，索引N-1）

    # 计算最大流并输出结果
    result = max_flow_solver.ford_fulkerson(source, sink)
    print(result)

main()
```

---

## I 研制能源

**题目复述**

科研人员正在实验室中研制一种全新的清洁的可再生能源，该新能源的
主要成分为两种物质:物质α和物质β。现在科研人员有n个烧杯，t毫升物质β，
第 i 个烧杯中现在有 li 毫升物质α，若研制成功将得到 pi 毫升的新能源，而科研
人员可以在烧杯中加入整数毫升的物质β，研制成功的概率为(物质β的量)/(物
质α的量+物质β的量)。现在科研人员想要合理分配他们的物质β使得研制成功
的新能源的量期望值最大，但是一个烧杯中的物质β不能超过物质α的量，否则
会发生爆炸。现在烧杯中物质α的量将会发生 q 次变动，科研人员想要知道每次
变动之后他们能获得的新能源的量的最大期望值

**测试样例**

*输入*
第一行三个正整数：n,t,q 
第二行 n 个正整数：pi 
第三行 n 个正整数：li 
接下来的 q 行每行两个正整数：tj，rj。tj 为 1 或 2，1 表示增加 1 毫升物质
α，2 表示减少 1 毫升物质α，rj 为变动的烧杯编号。保证任意时刻所有烧杯不
为空

*输出*
q 行实数表示每次变动之后的答案
如果绝对或相对误差不超过 1e-6，则答案将被视为正确。

*数据范围*
n,t,q<=200000，pi,li<=1000

*样例输入*
```
2 1 3 
4 5 
1 2 
1 1 
1 2 
2 1
```

*样例输出*
```
1.666666667 
1.333333333 
2.000000000
```

**样例解析**

1. **第一次变动：`1 1`**
   - **操作**：`tj = 1` 表示增加 1 毫升物质 α，`rj = 1` 表示第一个烧杯。
   - **更新后**：`l = [2, 2]`
   
   **分配方案分析：**
   
   可分配的物质 β 总量为 1 毫升，且每个烧杯中 `b_i ≤ l_i`。
   
   - **方案1**：将 1 毫升 β 分配给烧杯1 (`b1 = 1`, `b2 = 0`)
     $$
     \text{期望值} = \frac{1}{2 + 1} \times 4 + \frac{0}{2 + 0} \times 5 = \frac{1}{3} \times 4 + 0 = \frac{4}{3} \approx 1.333333333
     $$
   
   - **方案2**：将 1 毫升 β 分配给烧杯2 (`b1 = 0`, `b2 = 1`)
     $$
     \text{期望值} = \frac{0}{2 + 0} \times 4 + \frac{1}{2 + 1} \times 5 = 0 + \frac{1}{3} \times 5 = \frac{5}{3} \approx 1.666666667
     $$
   
   **最佳方案**：方案2，期望值为 `1.666666667`

2. **第二次变动：`1 2`**
   - **操作**：`tj = 1` 表示增加 1 毫升物质 α，`rj = 2` 表示第二个烧杯。
   - **更新后**：`l = [2, 3]`
   
   **分配方案分析：**
   
   - **方案1**：将 1 毫升 β 分配给烧杯1 (`b1 = 1`, `b2 = 0`)
     $$
     \text{期望值} = \frac{1}{2 + 1} \times 4 + \frac{0}{3 + 0} \times 5 = \frac{1}{3} \times 4 + 0 = \frac{4}{3} \approx 1.333333333
     $$
   
   - **方案2**：将 1 毫升 β 分配给烧杯2 (`b1 = 0`, `b2 = 1`)
     $$
     \text{期望值} = \frac{0}{2 + 0} \times 4 + \frac{1}{3 + 1} \times 5 = 0 + \frac{1}{4} \times 5 = \frac{5}{4} = 1.25
     $$
   
   **最佳方案**：方案1，期望值为 `1.333333333`

3. **第三次变动：`2 1`**
   - **操作**：`tj = 2` 表示减少 1 毫升物质 α，`rj = 1` 表示第一个烧杯。
   - **更新后**：`l = [1, 3]`
   
   **分配方案分析：**
   
   - **方案1**：将 1 毫升 β 分配给烧杯1 (`b1 = 1`, `b2 = 0`)
     $$
     \text{期望值} = \frac{1}{1 + 1} \times 4 + \frac{0}{3 + 0} \times 5 = \frac{1}{2} \times 4 + 0 = 2.0
     $$
   
   - **方案2**：将 1 毫升 β 分配给烧杯2 (`b1 = 0`, `b2 = 1`)
     $$
     \text{期望值} = \frac{0}{1 + 0} \times 4 + \frac{1}{3 + 1} \times 5 = 0 + \frac{1}{4} \times 5 = 1.25
     $$
   
   **最佳方案**：方案1，期望值为 `2.000000000`

**思路解析**

> 本思路和本代码在数据范围开满的情况下是过不了的, 暂未找到正确思路

每个烧杯分配 `b_i` 毫升的物质 β 后，研制成功的概率为：

$$
\text{概率} = \frac{b_i}{l_i + b_i}
$$

成功后得到的新能源量为 `p_i` 毫升，因此该烧杯的期望新能源量为：

$$
E_i = \frac{b_i}{l_i + b_i} \times p_i
$$

目标是分配 `t` 毫升的物质 β，使得所有烧杯的期望新能源量之和最大：

$$
\text{最大化} \quad \sum_{i=1}^{n} \frac{b_i}{l_i + b_i} \times p_i
$$

同时，需满足：

$$
0 \leq b_i \leq l_i \quad \text{且} \quad \sum_{i=1}^{n} b_i = t
$$

为了高效地选择当前边际增益最高的烧杯，可以采用优先级队列（最大堆）的数据结构。具体步骤如下：

1. **初始准备**：
   - 对于每个烧杯，初始化 `b_i = 0`。
   - 计算每个烧杯当前的边际增益：

     $$
     \text{Marginal Gain} = \frac{p_i \times l_i}{(l_i + b_i)^2} = \frac{p_i \times l_i}{(l_i)^2} = \frac{p_i}{l_i}
     $$

   - 将所有烧杯的初始边际增益及其相关信息（如 `i`、当前 `b_i`）加入最大堆。

2. **分配物质 β**：
   - 重复 `t` 次以下步骤：
     - 弹出堆顶元素，即当前边际增益最高的烧杯。
     - 将该烧杯的 `b_i` 增加 1（注意不能超过 `l_i`）。
     - 重新计算该烧杯的新边际增益，并将其重新加入堆中。

3. **处理动态变化**：
   - 每次烧杯中物质 α (`l_i`) 发生变化后，需要更新相关烧杯的边际增益。
   - 由于 `l_i` 变化影响到边际增益的计算，因此需要重新调整该烧杯在堆中的位置。
   - 具体步骤：
     - 更新对应烧杯的 `l_i`（增加或减少 1 毫升）。
     - 重新计算其当前的边际增益。
     - 更新堆中该烧杯的位置，确保堆的性质保持（即堆顶始终是当前最高的边际增益）。

**参考代码**

``` python
import heapq

def main():
    n, t, q = map(int, input().split())
    p = list(map(int, input().split()))
    l = list(map(int, input().split()))
    
    # Initialize b_i = 0 for all containers
    b = [0] * n
    
    # Compute initial marginal gains and build a max heap
    # Since heapq is a min heap, we use negative gains
    heap = []
    for i in range(n):
        if l[i] == 0:
            gain = 0
        else:
            gain = p[i] / (l[i] + 1)
        heap.append((-gain, i))
    
    heapq.heapify(heap)
    
    # Allocate t units
    for _ in range(t):
        if not heap:
            break
        neg_gain, i = heapq.heappop(heap)
        # Assign one unit to container i
        b[i] += 1
        if b[i] < l[i]:
            # Compute new marginal gain
            gain = p[i] * l[i] / (l[i] + b[i] + 1)**2
            heapq.heappush(heap, (-gain, i))
    
    # Compute initial expected value
    total_E = 0.0
    for i in range(n):
        if b[i] > 0:
            total_E += (b[i] / (l[i] + b[i])) * p[i]
    
    # Process queries
    for _ in range(q):
        tj, rj = map(int, input().split())
        rj -= 1  # 0-based index
        # Update l[rj]
        if tj == 1:
            l[rj] += 1
        elif tj == 2:
            l[rj] -= 1
        
        # Recompute the allocation
        # Reset allocations
        b = [0] * n
        heap = []
        for i in range(n):
            if l[i] == 0:
                gain = 0
            else:
                gain = p[i] / (l[i] + 1)
            heap.append((-gain, i))
        heapq.heapify(heap)
        # Allocate t units
        for _ in range(t):
            if not heap:
                break
            neg_gain, i = heapq.heappop(heap)
            # Assign one unit to container i
            b[i] += 1
            if b[i] < l[i]:
                # Compute new marginal gain
                gain = p[i] * l[i] / (l[i] + b[i] + 1)**2
                heapq.heappush(heap, (-gain, i))
        # Compute expected value
        total_E = 0.0
        for i in range(n):
            if b[i] > 0:
                total_E += (b[i] / (l[i] + b[i])) * p[i]
        # Print the result with 9 decimal places
        print("{0:.9f}".format(total_E))

main()
```

---

## J 清洁能源

**题目复述**

科研人员研制出了一种全新的清洁的可再生能源，这种可再生能源具
有非常神奇的性质。科研人员基于该可再生能源发明了一种发电装置，该装置有
n 个能源槽位，第 i 个能源槽位中放入了 i 份新能源。该发电装置的能源槽位的
位置可以进行随意交换，但是，对于其中任意一对能源槽位，只有满足当且仅当
这两个能源槽位的位置编号互质且其中的新能源量互质，这种情况下发电装置才
能得到最大的发电功率。
现在，发电装置的能源槽位已经被进行了一些交换，而其中一些能源槽位中
的新能源量已知，科研人员想要知道有多少种方案可以得到最大的发电功率，结
果对 1e9+7 取模。

**测试样例**

*输入*
输入包含多组数据，第一行输入数据组数 T 
每组数据的输入如下：
第一行一个正整数 n；第二行 n 个数 a1,a2,…,an；若 ai=0 则这个槽位的新能源量
未知，否则代表该槽位的新能源量

*输出*
满足条件的方案个数，对 1e9+7 取模。

*样例输入1*
```
1
4
0 0 0 0
```

*样例输出1*
```
4
```

*样例解析1*
满足的方案如下：
```
1 2 3 4
1 4 3 2
3 2 1 4
3 4 1 2
```

*样例输入2*
```
2 
5 
0 0 1 2 0 
6 
0 0 1 2 0 0
```

*样例输出2*
```
2
0
```

*样例解析2*
满足的方案数：
```
3 4 1 2 5
5 4 1 2 3
---
```

**题意概括**

数列${b_n} = {1, 2, ..., n}$ 的与数列 ${a_1, a_2, ..., a_n}$ 的数一一对应, 现在, 已知数列 ${a_n}$ 的一部分数, 
求有多少可能的情况使得$b_i, b_j$互质时, $a_i, a_j$ 也互质,
$b_i, b_j$不互质时, $a_i, a_j$ 也不互质.

**思路解析**

为了求解可行的方案数，我们需要搞清楚哪些数字和数字之间是等价的。
现在，我将解释哪些数之间是等价的（任何数字都和它自己等价）。

以7个槽位为例子，槽位的编号分别为 1，2，3，4，5，6，7
在这组数字中
2和4实际上是等价的，因为它只有2作为质因数
3只和它自己等价，因为3是质数，且编号中存在3的倍数（也就是6）
1和5和7是等价的，因为5和7是质数，且编号中不存在5或7的倍数，任何这样的数都和1是等价的
6只和自己等价，因为它有且仅有2和3这两个质因数，且编号中不存在任何同样的数

再以12个槽位为例，槽位编号分别为1，2，3，4，5，6，7，8，9，10，11，12
在这组数字中
1和7和11是等价的，因为11和7是质数，且编号中不存在11或7的倍数，任何这样的数都和1是等价的
2，4，8是等价的，因为他们都只有2这个质因数
3和9是等价的，因为他们都只有3这个质因数
6和12是等价的，因为他们都有且只有2和3这两个质因数
5只和自己等价，因为5是质数且编号中存在5的倍数
10只和自己等价，因为他有且只有2和5这两个质因数

当我们搞明白这一点以后，解决这道问题就容易了，我们只需要让等价的数之间任意交换，求出所有的可能性即可。
以如下样例为例：
输入：
```
1
5 
0 0 1 2 0 
```
输出：
```
2
```
可行方案：
```
3 4 1 2 5
5 4 1 2 3
```
在这个样例中，2和4等价，1和3和5等价
已经在编号4放置了2
则只能在编号2放置与其等价的4
已经在编号3放置了1
剩余编号1和编号5下待放置的3和5是等价的，故可以随意放置，
故共有如下两种放置方案：
```
3 4 1 2 5
5 4 1 2 3
```

**参考代码**

``` python
from math import gcd
from collections import defaultdict

MOD = 10**9 + 7

def get_prime_factors(x):
    """返回x的质因数集合"""
    if x == 1:
        return frozenset()
    factors = set()
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.add(d)
            x //= d
        d += 1
    if x > 1:
        factors.add(x)
    return frozenset(factors)

def identify_equivalence_classes(n):
    """
    根据质因数将1到n的数划分为等价类。
    返回一个字典，键为质因数集合，值为具有相同质因数的数列表。
    """
    classes = defaultdict(list)
    for x in range(1, n+1):
        factors = get_prime_factors(x)
        classes[factors].append(x)
    return list(classes.values())

def solve_case(n, a):
    """
    解决单个测试用例，返回满足条件的方案数。
    """
    # 识别等价类（目前未使用，可进一步优化）
    eq_classes = identify_equivalence_classes(n)
    
    # 确定已固定的槽位和可用的数
    fixed = {}
    available = set(range(1, n+1))
    for idx, val in enumerate(a):
        slot = idx + 1
        if val != 0:
            if val not in available:
                # 重复赋值，不可能
                return 0
            fixed[slot] = val
            available.remove(val)
    
    # 预处理槽位之间的互质关系
    coprime_slots = defaultdict(set)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i != j and gcd(i, j) == 1:
                coprime_slots[i].add(j)
    
    # 预处理数与数之间的互质关系
    # num_coprime[x]包含与x互质的所有数
    num_coprime = {}
    for x in range(1, n+1):
        num_coprime[x] = set()
        for y in range(1, n+1):
            if gcd(x, y) == 1:
                num_coprime[x].add(y)
    
    # 确定需要分配的槽位和可用数
    slots = list(range(1, n+1))
    slots_to_assign = [s for s in slots if s not in fixed]
    avail_numbers = list(available)
    
    # 如果槽位数大于可用数，返回0
    if len(slots_to_assign) > len(avail_numbers):
        return 0
    
    # 回溯法计数
    count = 0
    
    def backtrack(index, current_assignment, used):
        nonlocal count
        if index == len(slots_to_assign):
            count = (count + 1) % MOD
            return
        slot = slots_to_assign[index]
        for num in avail_numbers:
            if num not in used:
                # 检查与已分配的互质槽位
                valid = True
                for assigned_slot, assigned_num in current_assignment.items():
                    if assigned_slot in coprime_slots[slot]:
                        if gcd(num, assigned_num) != 1:
                            valid = False
                            break
                if valid:
                    # 选择num分配给slot
                    current_assignment[slot] = num
                    used.add(num)
                    backtrack(index + 1, current_assignment, used)
                    # 撤销选择
                    del current_assignment[slot]
                    used.remove(num)
    
    # 初始化当前分配为固定分配
    initial_assignment = fixed.copy()
    used_numbers = set(fixed.values())
    
    backtrack(0, initial_assignment, used_numbers)
    return count

def main():
    T = int(input())
    for _ in range(T):
        n = int(input())
        a = list(map(int, input().split()))
        result = solve_case(n, a)
        print(result)

main()
```

## K 公益活动

> K题样例格式有问题, 题面说ci可能小于0，pi一定大于0
> 但样例中ci全都大于0，pi却出现了小于0的情况
> 故暂时没法做
