<!--
<details> 
<summary> Code </summary>

```cpp

```

</details>
-->


**全称 AtCoder Educational DP Contest，是一个高质量的 DP 题单，题目较为基础，平均难度为绿，很多题都是练习基础 DP 不可多得的经典好题。**

题单链接：[Link](https://www.luogu.com.cn/training/244301)

题目编号格式为 `AT_dp_?`，`?` 为小写字母。题单共 $26$ 题。

题目 A 到 N 较基础，O 到 Z 为进阶内容。

----

## A. Frog 1

【题目大意】

$N$ 个石头，编号为 $1,2,...,N$。对于每个 $i$（$1 \leq i \leq N$），石头 $i$ 的高度为 $h_i$。

最初有一只青蛙在石头 $1$ 上。他将重复几次以下操作以到达石头 $N$：

- 如果青蛙当前在石头 $i$ 上，则跳到石头 $i+1$ 或石头 $i+2$。需要 $\lvert h_i - h_j \rvert$ 的费用，而 $j$ 是要落到上面的石头。

找到青蛙到达石头 $N$ 之前需要的最小总费用。

- $2 \leq N \leq 10^5$，$1 \leq h_i \leq 10^4$。

【解题思路】

设 $f_i$ 为跳到第 $i$ 块石头的最小花费，设 $d_{i, j} = \lvert h_i - h_j \rvert$，容易得到状态转移方程

$$
f_i = \min\{f_{i-1} + d_{i,i-1}, f_{i-2} + d_{i,i-2}\}
$$

注意此时 $i > 2$。当 $i = 2$ 时，$f_i = d_{1, 2}$。同时，其他的 $f_i$ 应该初始化为无穷大。

当然，这题也可以建图，给 $i,i+1$ 和 $i,i+2$ 连边然后跑 $\texttt{Dijkstra}$。

----

## B. Frog 2

【题目大意】

河面上有 $N$ 块石头。有一只青蛙在第 $1$ 块石头上，它想跳到第 $N$ 块石头上。

青蛙一次最多只能跳过 $K$ 块石头。从第$i$块跳到第$j$块需要花费青蛙 $\lvert h_i - h_j \rvert$ 的体力。求青蛙到达第$N$块石头所耗费的最小体力值。

- $2 \leq N \leq 10^5$，$1 \leq K \leq 100$，$1 \leq h_i \leq 10^4$。

【解题思路】

跟上一题一样，只不过 $f_i$ 不只是由 $f_{i-1}$ 和 $f_{i-2}$ 转移而来，而是由 $f_{i-1},f_{i-2},f_{i-3} \cdots f_{i-k}$ 转移过来。这样，容易得到状态转移方程

$$
f_i = \min_{\max\{0, i-k\} \leq j < i} \{f_j + d_{i, j}\}
$$

需要注意数组下标不能小于 $0$。同时 $f_i$ 初始为无穷大，且 $f_0 = 0$。

----

## C. Vacation

【题目大意】

暑假有 $N$ 天。对于每一天 $i$（$1 \leq i \leq N$），太郎君可以选择以下活动之一：

- A：在海里游泳，获得幸福度 $a_i$。
- B：在山上抓虫，获得幸福度 $b_i$。
- C：在家做作业，获得幸福度 $c_i$。

由于太郎君容易厌倦，他不能连续两天及以上做同样的活动。

请计算太郎君可以获得的最大总幸福度。

- $1 \leq N \leq 10 ^ 5$，$1 \leq a _ i, b _ i, c _ i \leq 10 ^ 4$。

【解题思路】

设 $f_{i, j}$ 表示，前 $i$ 天，第 $i$ 天选取活动 $j$ 的最大总幸福值。为了方便，我们将这些活动的幸福值都拿一个二维数组存储，即 $a_{i, j}$ 表示第 $i$ 天选取活动 $j$ 的幸福值。这里，都有 $1 \leq j \leq 3$。

由于相邻的天不能选取同一种活动，$f_{i, 1}$ 只能由 $f_{i-1,2}$、$f_{i-1,3}$ 转移过来，则有

$$
f_{i,1} = \max\{f_{i-1,2}, f_{i-1,3}\} + a_{i, 1} 
$$

同理，有

$$
f_{i,2} = \max\{f_{i-1,1}, f_{i-1,3}\} + a_{i,2}
$$

且

$$
f_{i,3} = \max\{f_{i-1,1}, f_{i-1,2}\} + a_{i,3} 
$$

答案即为 $\max\{f_{n,1}, f_{n, 2}, f_{n, 3}\}$。

----

## D. Knapsack 1

【题目大意】

$N$ 个物品，选取其中若干个物品，使得对选取的这些物品 $\sum w_i\leq W$ 的前提下最大化 $\sum v_i$。  

其中，$w_i$ 为重量，$v_i$ 为价值。

- $1 \leq N \leq 100$，$1 \leq w_i \leq W \leq 10^5$，$1 \leq v_i \leq 10^9$。

【解题思路】

01 背包板子题。设 $f_{i, j}$ 表示前 $i$ 个物品，用了 $j$ 容量的最大价值和。则有

$$
f_{i, j} = \max\{f_{i-1, j-w_i}+v_i, f_{i-1, j}\}
$$

答案为 $f_{N, W}$。

这个方程空间是二维的，考虑优化。

我们注意到，第一维（即 $i$）只由 $i-1$ 转移过来，那么 $i-1$ 前面的都与 $i$ 这维无关，而我们的答案是 $f_{N, W}$，其实我们不需要前面这些东西。

那么怎么办？我们可以直接省略掉第一维。这样，我们在计算 $i$ 的值之前，这个 $f$ 数组其实表示的就是 $i-1$ 对应值。这样，我们就有转移方程

$$
f_j = \max\{f_j, f_{j-w_i} + v_i\}
$$

这种优化方法叫做**滚动数组**。

注意一个小细节：在枚举 $j$ 时候，需要倒序循环，为什么？我们在转移时，$f_{j-w_i}$ 表示的是原来的 $f_{i-1,j-w_i}$。但是，这个性质成立有个条件：$f_{j-w_i}$ 没被计算过。否则它表示的就是 $f_{i, j-w_i}$。如果正序循环，它正好被计算过，所以表示的是 $f_{i,j-w_i}$，这样显然是不行的。倒序循环就能很好地避免这个问题。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long 
using namespace std;

const int N = 1e5 + 10;
int n, m;
int w[N], v[N];
int f[N];

signed main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> w[i] >> v[i];
    }
    for (int i = 1; i <= n; i++) {
        for (int j = m; j >= w[i]; j--) {
            f[j] = max(f[j], f[j - w[i]] + v[i]);
        }
    }
    cout << f[m];
    return 0;
}
```

</details>

----

## E. Knapsack 2

【题目大意】

与上一题一样，只不过 $W$ 的最大值变为 $10^9$，$v_i$ 的最大值变为 $v_i$。

【解题思路】

注意到 $v_i$ 的范围很小，那么我们只需要将状态变为 $f_{j}$ 表示选取**价值**为 $j$ 时的最小**重量**和。

转移方程与上一题完全一样（当然，要把 $v_i$ 和 $w_i$ 交换）。

如何求答案？枚举 $1$ 到最大价值（即价值和），合法（即 $f_i \leq W$）就输出。倒序循环。

-----

## F. LCS

【题目大意】

给定一个字符串 $s$ 和一个字符串 $t$ ，输出 $s$ 和 $t$ 的最长公共子序列。

- $s$ 和 $t$ 的长度均小于 $3000$。

【解题思路】

设 $f_{i, j}$ 表示 $s$ 的前 $i$ 个字符，$t$ 的前 $j$ 个字符的最长公共子序列。则有

$$
f_{i, j} = \begin{cases}f_{i-1,j-1}+1 & s_i=t_j\\
\max\{f_{i-1,j}, f_{i,j-1}\} & s_i\neq t_j
\end{cases}
$$

但是这题需要输出最长公共子序列。显然，在第一种转移中对最长公共子序列有贡献，那么只需要在第一种转移时记录一下字符即可。

----

## G. Longest Path

【题目大意】

求**有向无环图**上的最长路长度。

长度为路径上边的数量。

【解题思路】

注意到是 DAG（有向无环图），考虑拓扑排序。

对于边 $(u, v)$，有 $d_v = d_u + 1$。答案为 $d_i$ 的最大值。

----

## H. Grid 1

【题目大意】

给一个 $H\times W$ 的网格，一开始在左上角 $(1,1)$ 每一步只能向右或向下走，不能经过 `#` 格子，求走到右下角 $(H,W)$ 有多少种走法。  

答案对 $10^9+7$ 取模。

- $2 \leq H, W \leq 1000$。

【解题思路】

直接二维 DP 就行，不讲了。

----

## I. Coins

【题目大意】

设 $N$ 是一个正的奇数。

有 $N$ 枚硬币，每枚硬币上标有编号 $1, 2, \ldots, N$。对于每个 $i$ ($1 \leq i \leq N$)，掷硬币 $i$ 时，正面朝上的概率是 $p _ i$，反面朝上的概率是 $1 - p _ i$。

太郎君把这 $N$ 枚硬币全部投掷了一次。请计算正面朝上的硬币数多于反面朝上的硬币数的概率。

- $1 \leq N \leq 1000$。

【解题思路】

设 $f_{i, j}$ 为前 $i$ 个硬币，有 $j$ 个朝上的概率。考虑第 $i$ 个硬币的朝向，则有

$$
f_{i, j} = f_{i-1, j-1} \times p_i + f_{i-1,j}\times (1-p_i)
$$

记答案为 $Ans$，有

$$
Ans = \sum_{i=\lceil\frac{n}{2}\rceil}^{n} f_{n, i}
$$

其实这题可以滚动数组将空间复杂度优化为 $O(n)$，留作思考内容。

其实这题还可以使用分治 FFT 将时间复杂度优化为 $O(n \log ^2 n)$，但是我不会，留作思考内容。 

----

## J. Sushi

【题目大意】

现有 $N$ 个盘子，编号为 $1,2,3,…,N$。第 $i$ 个盘子中放有 $a_i$ 个寿司。

接下来每次执行以下操作，直至吃完所有的寿司。从第 $1,2,3,…,N$ 个盘子中任选一个盘子，吃掉其中的一个寿司。若没有寿司则不吃。

若将所有寿司吃完，请问此时操作次数的数学期望是多少？

- $1 \leq N \leq 300$，$1 \leq a_i \leq 3$。

【解题思路】

期望 DP，先咕了。

----

## K. Stones

【题目大意】

$N$ 个正整数组成的集合 $A = \{ a _ 1, a _ 2, \ldots, a _ N \}$。太郎君和次郎君将用以下游戏进行对决。

首先，准备一个有 $K$ 个石子的堆。两人依次进行以下操作。太郎君先手。

- 从集合 $A$ 中选择一个元素 $x$，从石堆中恰好移除 $x$ 个石子。

不能进行操作的人输掉游戏。当两人都按照最优策略行动时，判断谁会获胜。

- $1 \leq N \leq 100$，$1 \leq K \leq 10^5$，$ 1\leq a_1 < a_2 < \cdots < a_N \leq K $。


【解题思路】

博弈论，先咕了。

----

## L. Deque

【题目大意】

给一个长度为 $N$ 的双端队列，第 $i$ 个数为 $a_i$。双方轮流取数，每一次能且只能从队头或队尾取数，取完数后将这个数从队列中弹出。双方都希望自己取的所有数之和尽量大，且双方都以最优策略行动，假设先手取的所有数之和为 $X$，后手取的所有数之和为 $Y$，求 $X-Y$。

- $1 \leq N \leq 3000$，$1 \leq a_i \leq 10^9$。

【解题思路】

区间 DP。

设 $f_{l, r}$ 表示区间 $[l, r]$ 的最终分数差。我们可以通过 $[l, r]$ 的长度来判断这次是由谁取，于是可以从 $[l + 1, r]$ 或者 $[l, r - 1]$ 转移过来。

注意区间 DP 需要先枚举区间长度再枚举区间两端点。否则可能在计算 $[l, r]$ 时并没有计算过 $[l + 1, r]$ 或 $[l, r - 1]$。

具体做法见代码。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long 
using namespace std;

const int N = 3010;
int n, a[N];
int f[N][N];

signed main() {
    cin >> n;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    for (int le = 1; le <= n; le++) {
        for (int l = 1; l + le - 1 <= n; l++) {
            int r = l + le - 1;
            if ((n - le) & 1) f[l][r] = min(f[l + 1][r] - a[l], f[l][r - 1] - a[r]);
            else f[l][r] = max(f[l + 1][r] + a[l], f[l][r - 1] + a[r]);
        }
    }
    cout << f[1][n];
    return 0;
}
```

</details>

----

## M. Candies

【题目大意】

$K$ 颗糖分给 $N$ 个人，第 $i$ 个人至少分得 $0$ 颗，至多分得 $a_i$ 颗，必须分完，求方案数，答案对 $10^9+7$ 取模。

- $1 \leq N \leq 100$，$1 \leq K \leq 10^5$，$1 \leq a_i \leq K$。

【解题思路】

设 $f_{i, j}$ 表示前 $i$ 个人，已经用掉 $j$ 个糖果的方案数，那么有

$$
f_{i, j} = \sum_{k = \max(0, j-a_i)}^{K} f_{i-1,k}
$$

这个式子直接转移的复杂度是 $O(n K^2)$ 的，显然不能接受。

观察这个式子，我们发现，后面那个 $\sum$ 是一个连续段的求值，可以直接对 $f_{i - 1, j}$ 做前缀和。这样我们可以做到 $O(1)$ 转移，总时间复杂度 $O(nK)$。

具体做法见代码。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long 
using namespace std;

const int N = 110;
const int K = 1e5 + 10;
const int P = 1e9 + 7;
int n, k, a[N];
int f[N][K], s[K];

signed main() {
    cin >> n >> k;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
    }
    f[0][0] = 1ll;
    for (int i = 1; i <= n; i++) {
        s[0] = f[i - 1][0];
        for (int j = 1; j <= k; j++) {
            s[j] = (s[j - 1] + f[i - 1][j]) % P;
        }
        for (int j = 0; j <= k; j++) {
            int p = max(0ll, j - a[i]);
            if (p == 0) f[i][j] = s[j];
            else f[i][j] = (s[j] - s[p - 1] + P) % P;
        }
    }
    cout << f[n][k];
    return 0;
}
```

</details>

----

## N. Slimes

【题目大意】

有 $N$ 个数，第 $i$ 个数是 $a_i$ ，现在要进行 $N-1$ 次操作。

对于每一次操作，可以把相邻两个数合并起来，并写上他们的和，这次操作的代价就是这个和。

求代价最小值。

- $2 \leq N \leq 400$，$1 \leq a_i \leq 10^9$。

【解题思路】

区间 DP。

设 $f_{l, r}$ 表示合并区间 $[l, r]$ 的最小代价。枚举区间内的“断点” $k$，有

$$
f_{l, r} = \min_{l \leq k < r} \{f_{l, k} + f_{k + 1, r} + \sum_{i = l}^r a_i\}
$$

直接枚举即可。

----

## O. Matching

【题目大意】

给定二分图，两个集合都有 $N$ 个点，$a_{i,j}=1$ 表示第一个集合第 $i$ 个点与第二个集合第 $j$ 个点连边。

求二分图完备匹配数，答案对 $10^9+7$ 取模。

- $1 \leq N \leq 21$。

【解题思路】

令符号 $|$ 表示按位或运算。

观察数据范围可知，这题是状压 DP。

令 $f_{i, j}$ 表示第一个集合前 $i$ 个点**全部**匹配完，此时第二个集合匹配状态为 $j$ 时，匹配的数量。

这里，状态 $j$ 表示：如果在二进制下，$j$ 的第 $t$ 位为 $1$，说明第二个集合的点 $t$ 被匹配了，否则未匹配。 

容易知道，二进制下，$j$ 中 $1$ 的数量一定为 $i$，否则匹配不成立。统计二进制下 $1$ 的个数可以使用 `__builtin_popcount()`。


考虑由 $f_{i, j}$ 转移到 $f_{i + 1, ?}$，枚举第 $i + 1$ 个点匹配的点 $k$，那么状态变为 $j | 2^k$。即

$$
f_{i, j} \to f_{i+1, j | 2^k} (a_{i, k} = 1)
$$

具体实现见代码。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
using namespace std;

const int N = 22;
const int M = (1 << N);
const int P = 1e9 + 7;
int n, a[N][N];
int f[N][M];

int main() {
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> a[i][j];
        }
    }
    f[0][0] = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < (1 << n); j++) {
            if (__builtin_popcount(j) != i) continue;
            for (int k = 0; k < n; k++) {
                if (!a[i][k]) continue;
                f[i + 1][j | (1 << k)] = (f[i + 1][j | (1 << k)] + f[i][j]) % P;
            }
        }
    }
    cout << f[n][(1 << n) - 1];
    return 0;
}
```

</details>

----

## P. Independent Set

【题目大意】

给一棵 $N$ 个点的树，每一个点可以染成黑色或白色，任意两个相邻节点不能都是黑色，求方案数，结果对 $10^9+7$ 取模。

- $1 \leq N \leq 10^5$。

【解题思路】

记集合 $S_u$ 表示点 $u$ 的子节点。

设 $f_{i, 0} / f_{i, 1}$ 分别表示第 $i$ 个点染成白色/黑色的方案数。容易得到状态转移方程

$$
f_{u, 0} = \prod_{v \in S_u} f_{v, 0} + f_{v, 1} 
$$

和

$$
f_{u, 1} = \prod_{v \in S_u} f_{v, 0}
$$

在 DFS 的时候转移即可。具体实现见代码。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long 
using namespace std;

const int N = 1e5 + 10;
const int P = 1e9 + 7;
int n, f[N][2];
int h[N], tot;

struct edge{
    int to, nxt;
}e[N << 1];

void add(int u, int v) {
    e[++tot] = {v, h[u]};
    h[u] = tot;
}
void dfs(int u, int fa) {
    f[u][0] = f[u][1] = 1;
    for (int i = h[u]; i; i = e[i].nxt) {
        int v = e[i].to;
        if (v == fa) continue;        
        dfs(v, u);
        f[u][0] = (f[u][0] * ((f[v][1] + f[v][0]) % P)) % P;
        f[u][1] = (f[u][1] * f[v][0]) % P;
    }
}

signed main() {
    scanf("%lld", &n);
    for (int i = 1; i < n; i++) {
        int u, v;
        scanf("%lld%lld", &u, &v);
        add(u, v), add(v, u);
    }
    dfs(1, 0);
    cout << (f[1][0] + f[1][1]) % P;
    return 0;
}
```

</details>

----

## Q. Flowers

【题目大意】

有一排花，共 $N$ 个，第 $i$ 个的高度是 $h_i$ ，权值是 $a_i$ ，保证高度互不相同。现在拿走一些花，使剩下的花高度单调递增，问剩下的花权值之和最大是多少。

- $1 \leq h_i \leq N \leq 2 \times 10^5$，$1 \leq a_i \leq 10^9$。

【解题思路】

设 $f_i$ 表示对于前 $i$ 个花原问题的答案。容易写出状态转移方程

$$
f_i = \max_{1 \leq j < i \land h_j < h_i} f_j + a_i
$$

这个式子直接转移是 $O(n^2)$ 的，考虑优化。

注意到使用原来的数组下标似乎不好维护，我们可以使用 $h_i$ 为下标。具体地，维护一个序列，下标为 $h_i$，值为 $f_i$。这样我们只需要查找下标为 $1$ 到 $h_i$ 的最大值即可。容易用树状数组维护。

----

## R. Walk

【题目大意】

给一张有 $N$ 个点的有向简单图，给出邻接矩阵，求长度为 $K$ 的路径条数，答案对 $10^9 + 7$ 取模。

- $1 \leq N \leq 50$，$1 \leq K \leq 10^{18}$。

【解题思路】

设 $f_{u, v}^i$ 表示 $u \to v$ 且长度为 $i$ 的路径数量，则有

$$
f_{u, v}^i = \sum_{k = 1}^n f_{u, k}^{i-1} \times f_{k, v}^{i-1}
$$

答案为

$$
\sum_{i = 1}^n \sum_{j = 1}^n f_{i, j}^K
$$

注意到第一个式子与矩阵乘法的式子完全一样，矩阵快速幂即可。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long
using namespace std;

const int M = 60;
const int P = 1e9 + 7;
int n, k, a[M][M];

struct matrix{
    int a[M][M];
    matrix() {memset(a, 0, sizeof a);}
    matrix operator * (const matrix &b) const {
        matrix res;
        for (int i = 1; i < M; i++) {
            for (int j = 1; j < M; j++) {
                for (int k = 1; k < M; k++) {
                    res.a[i][j] = (res.a[i][j] + a[i][k] * b.a[k][j] % P) % P;
                }
            }
        }
        return res;
    }
}ans, bs;

void qp(int b) {
    for (int i = 1; i <= n; i++) {
        ans.a[i][i] = 1;
    }
    while (b) {
        if (b & 1) ans = ans * bs;
        bs = bs * bs;
        b >>= 1;
    }
}

signed main() {
    cin >> n >> k;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            cin >> bs.a[i][j];
        }
    }
    qp(k);
    int la = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            la = (la + ans.a[i][j]) % P;
        }
    }
    cout << la;
    return 0;
}
```

</details>

----

## S. Digit Sum

【题目大意】

$1$ 和 $K$ 之间有多少个整数，使得十进制表示的数字之和是 $D$ 的倍数？答案对 $10^{9} + 7$ 取模。

- $1 \leq K \leq 10^{10^4}，1 \leq D \leq 100$。

【解题思路】

数位 DP。一般使用记忆化搜索。

状态有三个参数 $i, j, lm$，分别表示搜索的位数、数字之和除以 $D$ 的余数，是否到数字的上界。

具体实现见代码。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long 
using namespace std;

const int N = 10010;
const int D = 110;
const int P = 1e9 + 7;
int n, d, a[N];
string k;
int f[N][D];

int dfs(int i, int j, int lm) {
    if (!lm && f[i][j] != -1) return f[i][j];
    if (!i) return (!j);
    int ans = 0, mx = (lm ? a[i] : 9);
    for (int k = 0; k <= mx; k++) {
        ans = (ans + dfs(i - 1, (j + k) % d, lm && (k == mx))) % P;
    } 
    if (!lm) f[i][j] = ans;
    return ans;
}
int solve(string s) {
    memset(f, -1, sizeof f);
    int cnt = s.size();
    for (int i = 0; i < cnt; i++) {
        a[i + 1] = s[cnt - i - 1] - '0';
    }
    return dfs(cnt, 0, 1);
}

signed main() {
    cin >> k;
    cin >> d;
    cout << (solve(k) % P - 1 + P) % P;
    return 0;
}
```

</details>

----

## T. Permutation

【题目大意】

有一个长为 $N$ 的正整数排列。给定一个由 `<` 和 `>` 组成长为 $N-1$ 的的字符串。

对于任意满足 $1 \le i \le N-1$ 的字符 $s_i$，如果 $s_i$ 是 `<` 则 $P_i<P_{i+1}$、如果 $s_i$ 是 `>` 则 $P_i>P_{i+1}$。求满足这样的性质的排列 $P$ 的方案数。答案对 $10^9 + 7$ 取模。

【解题思路】

设 $f_{i, j}$ 表示在前 $i$ 个位置为 $1\sim i$ 的排列，且第 $i$ 个位置为 $j$ 的方案数。

考虑由 $f_{i-1, ?}$ 转移到 $f_{i, j}$。我们可以枚举第 $i - 1$ 个位置填什么，然后将前面排列中 $\geq j$ 的数全部加一即可。

具体地，若 $s_i$ 为 `<`，有

$$
f_{i, j} = \sum_{k = 1}^{j - 1} f_{i - 1, k}
$$

若 $s_i$ 为 `>`，有
 
$$
f_{i, j} = \sum_{k = j}^{i - 1} f_{i - 1, k}
$$

观察转移式，我们发现 $f_{i, ?}$ 仅由 $f_{i - 1, ?}$ 转移过来，而且是一个连续段的求和，前缀和优化即可。

----

## U. Grouping

【题目大意】

对 $N$ 个物品任意分组，如果第 $i$ 个物品和 第 $j$ 个物品分在一组，会产生 $a_{i,j}$ 的得分，最大化得分之和。  

$(i,j)$ 和 $(j,i)$ 的贡献只计算一次。

- $1 \leq N \leq 16$，$\lvert a_{i, j} \rvert \leq 10^9$。

【解题思路】

考虑状压 DP。

设 $f_S$ 表示已选集合为 $S$ 时，最大的得分和。

若只将 $S$ 分为一部分，有

$$
f_S = \sum_{u, v \in S \land u < v} a_{u, v}
$$

容易写出状态转移方程

$$
f_S = \max_{T \subseteq S} \{f_T + f_{\complement_S T}\}
$$

直接状压即可。

----

## V. Subtree

【题目大意】

给一棵 $N$ 个节点的树，对每一个节点染成黑色或白色。  

对于每一个节点，求强制把这个节点染成黑色的情况下，所有的黑色节点组成一个联通块的染色方案数，答案对 $M$ 取模。

- $1 \leq N \leq 10^5$，$1 \leq M \leq 10^9$。

【解题思路】

换根 DP。

记集合 $S_u$ 表示 $u$ 的子节点。点 $F_u$ 表示 $u$ 的父节点。 

设 $f_u$ 为 $u$ 为黑色时，$u$ 的子树**内**黑色点组成一个联通块的个数。$g_u$ 为 $u$ 为黑色时，$u$ 的子树**外**黑色点组成一个联通块的个数。

容易写出状态转移方程

$$
f_u = \prod_{v \in S_u} (f_v+1)
$$

和

$$
g_u = g_{F_u} \times \prod_{v \in S_{F_u} \land v\neq u} (g_v + 1) + 1
$$

答案即为

$$
\max_{1 \leq i \leq N} \{ f_i \times g_i \}
$$

考虑优化第二个转移式。注意到这是两个连续段的乘积，维护一个前缀积、一个后缀积即可。

----

## W. Intervals

【题目大意】

给定 $M$ 条规则形如 $(l_i,r_i,a_i)$，对于一个 01 串，其分数的定义是：对于第 $i$ 条规则，若该串在 $[l_i,r_i]$ 中至少有一个 1，则该串的分数增加 $a_i$。

你需要求出长度为 $N$ 的 01 串中的最大分数。

- $1 \leq N,M \leq 2 \times 10^5$，$\lvert a_i \rvert \leq 10^9$。

【解题思路】

线段树优化 DP，比较套路，让我先研究一会。

----

## X. Tower

【题目大意】

你有 $N$ 个箱子，编号从 $1$ 到 $N$，每个箱子有三个属性，以第 $i$ 个箱子为例，分别是重量 $w_i$，承重能力 $s_i$，价值 $v_i$。

你想建一座塔，因此需要将一些箱子堆叠起来，但是每个箱子必须满足下面的条件：

- 这个箱子上面的所有箱子重量和要小于等于这个箱子的承重能力。

定义一个塔的价值为它所用的所有箱子的价值和。

最大化这个塔的价值并输出它。

- $1 \leq N \leq 10^3$，$1 \leq w_i, s_i \leq 10^4$，$1 \leq v_i \leq 10^9$。

【解题思路】

咕。

----

## Y. Grid 2

【题目大意】

给一个 $H\times W$ 的网格，每一步只能向右或向下走，给出 $N$ 个坐标 $(r_1,c_1),(r_2,c_2),...,(r_n,c_n)$，这些坐标对应的位置不能经过，求从左上角 $(1,1)$ 走到右下角 $(H,W)$ 的方案数。答案对 $10^9+7$ 取模。

- $2 \leq H, W \leq 10^5$，$1 \leq N \leq 3000$。

【解题思路】

注意到 $N$ 的范围较小，考虑从障碍物入手。

!!! note "引理"
    从点 $(x_1, y_1)$ 走到 $(x_2, y_2)$，一共有 ${x_2-x_1+y_2-y_1 \choose x_2-x_1}$ 种走法。

??? note "证明"
    一共需要走 $x_2-x_1+y_2-y_1$ 步，其中需要选取 $x_2-x_1$ 步走在 $x$ 轴方向上。

设 $f_i$ 为从 $(1,1)$ 走到第 $i$ 个障碍物的方案数。为了方便处理，记点 $(H,W)$ 为第 $N+1$ 个障碍物，那么答案即为 $f_{N+1}$。

如何转移？考虑容斥。$f_i$ 即为从 $(1,1)$ 走到 $(r_i,c_i)$ 的方案数（不考虑障碍物），减去至少经过一个障碍物的方案数。

显然，只有这个点左上角的障碍物才会对 $f_i$ 产生影响，枚举障碍物转移即可。

具体地，有状态转移方程

$$
f_i = {r_i+c_i-2\choose r_i-1} - \sum_{1\leq j\leq N \land i\neq j \land r_j\leq r_i \land c_j\leq c_i}f_j+{r_i-r_j+c_i-c_j \choose c_i-c_j}
$$

注意先对点排序，以确保转移顺序合法。具体实现见代码。

<details> 
<summary> Code </summary>

```cpp
#include <bits/stdc++.h>
#define int long long 
using namespace std;

const int N = 2e5 + 10;
const int P = 1e9 + 7;
int h, w, n;
int f[N];
int fac[N], inv[N];

struct node{
    int x, y;
    bool operator < (const node &tx) const {
        if (x == tx.x) return y < tx.y;
        return x < tx.x;
    }
}a[N];

int qp(int a, int b) {
    int as = 1;
    while (b) {
        if (b & 1) {
            as = (as * a) % P;
        }
        a = (a * a) % P;
        b >>= 1;
    }
    return as % P;
}
int C(int n, int m) {
    if (n < m) return 0;
    return fac[n] * inv[m] % P * inv[n - m] % P;
}

signed main() {
    cin >> h >> w >> n;
    for (int i = 1; i <= n; i++) {
        cin >> a[i].x >> a[i].y;
    }
    fac[0] = inv[0] = 1;
    for (int i = 1; i < N; i++) {
        fac[i] = fac[i - 1] * i % P;
        inv[i] = qp(fac[i], P - 2) % P;
    }
    sort(a + 1, a + n + 1);
    a[n + 1].x = h, a[n + 1].y = w;
    f[0] = 1ll;
    for (int i = 1; i <= n + 1; i++) {
        int tx = a[i].x, ty = a[i].y;
        f[i] = C(tx + ty - 2, tx - 1) % P;
        for (int j = 1; j < i; j++) {
            int nx = a[j].x, ny = a[j].y;
            if (nx > tx || ny > ty) continue;
            f[i] = (f[i] - (f[j] * C(tx - nx + ty - ny, tx - nx)) % P + P) % P;
        }
    }
    cout << f[n + 1] % P;
    return 0;
}
```

</details>

----

## Z. Frog 3

【题目大意】

有 $N$ 个石头，编号为 $1,2,\dots,N$，第 $i$ 个高为 $h_i$。保证 $h$ 严格单调递增。

有一只青蛙在第一个石头上，它可以跳到石头编号为 $i+1,i+2,\dots,N$。当他跳到编号 $j$ 石头时的花费是
$(h_i-h_j)^2+C$。求跳到编号为 $N$ 石头的最小花费。

- $2 \leq N \leq 2 \times 10^5$，$1 \leq C \leq 10^{12}$，$1 \leq h_1 < h_2 < \cdots < h_N \leq 10^6$。

【解题思路】

斜率优化，写个学习笔记再更。

