## 前言

后缀数组板子一遍过了，开心。

这篇学习笔记整体框架上与 OI-Wiki 相似，但保证本文章大部分原创。

前置知识：基数排序、倍增。

## 记号与约定

字符串即为 $s$。

记 "后缀 $i$" 表示从 $i$ 开始的后缀，即 $s[i\dots n]$，它代表后缀的编号。

记 $sa[i]$ 表示排名为 $i$ 的后缀的编号，$rk[i]$ 表示后缀 $i$ 的排名。显然有 $sa[rk[i]] = rk[sa[i]] = i$。

剩下的数组下文会说。

## 求法

### $O(n^2\log n)$ 做法

暴力对每个后缀进行排序，每次比较的复杂度为 $O(n)$，排序复杂度为 $O(n\log n)$，总复杂度 $O(n^2\log n)$。

### $O(n \log^2{n})$ 做法

对于字符串 $s$ 的所有后缀，它们有大量重复部分，而直接排序就会进行大量重复比较，不如换个角度入手。

我们先从一组数据入手。对于 $s = \texttt{abbaaaba}$，我们先以每个后缀第一个字符为第一关键字，第二个字符为第二关键字进行排序，其实也就是对于每个后缀的前两个字符进行排序，结果如下：

![](images/img1.png)

根据上面的定义，绿色部分的排名即为 $rk$ 数组。观察这些后缀，我们现在知道绿色部分的排名。而由后缀的性质，蓝色部分其实也是后缀，那么其实我们也知道下图中黄色部分的排名：

![](images/img2.png)

那么我们使用绿色部分的排名（即原排名，$rk$ 数组）为第一关键字，黄色部分的排名（即对于后缀 $i$，第一个图中的蓝色部分是后缀 $i + 2$，那么黄色部分的排名即为后缀 $i + 2$ 的原排名，即 $rk[i + 2]$）为第二关键字，进行排序，结果如下：

![](images/img3.png)

很好，现在绿色部分（已排序部分）的长度由 $2$ 变为了 $4$。我们接下来再选取长度为 $4$ 的黄色部分，这样，我们就成功对后缀的前 $8$ 个字符排序了。以此类推，我们每次都倍增排序长度，并且重复以上操作，那么我们就可以在 $O(n \log^2{n})$ 的复杂度内解决掉这个问题了（排序 $O(n\log n)$，倍增 $O(\log n)$）。

???+ note "形式化的解法"
    设需要排序长度为 $w$。
    
    对于后缀 $i$，我们知道按它的第 $1$ 个字符到第 $w/2$ 个字符排序的排名。对于后缀 $i + w/2$ 也是如此，这相当于我们知道按后缀 $i$ 的第 $w/2 + 1$ 个字符到第 $w$ 个字符排序的排名。那么我们以 $rk[i]$ 为第一关键字，$rk[i+w/2]$ 为第二关键字进行排序即可。

    倍增 $w$，即 $w \gets 2 \times w$。

### $O(n \log{n})$ 做法

字符串有一个特点：值域小。也就是说每个位置的取值种类少。这样，我们就可以利用基数排序的思想。

具体地，先按第一个关键字扔进桶里，再按从大到小按第二关键字遍历桶即可。代码中的第二关键字进行了离散化，它代表第二关键字的排名。代码如下：

```cpp
void f_sort() {
    for (int i = 1; i <= m; i++) b[i] = 0;
    for (int i = 1; i <= n; i++) b[rk[i]]++;
    for (int i = 1; i <= m; i++) b[i] += b[i - 1];
    for (int i = n; i >= 1; i--) sa[b[rk[tp[i]]]--] = tp[i];
}
```

可以看到，这样排序的复杂度是 $O(n)$。我们成功地将总复杂度优化为 $O(n\log n)$。

我们求出 $sa$ 数组了之后，不能直接用 $rk[sa[i]] = i$ 给 $rk$ 赋值，这是因为有可能有些后缀的第一二关键字都相同，那么它们的排名也相同，这需要特判一下。

容易观察到第二关键字是由第一关键字平移得到，那么我们就可以 $O(n)$ 求第二关键字。而且第二关键字我们只需要排名，可以进行一点小优化。具体实现见代码。

??? note "完整代码"
    ```cpp
    #include <bits/stdc++.h>
    using namespace std;

    const int N = 1e6 + 10;
    int n, m;
    int sa[N], rk[N];
    int b[N], tp[N];
    char c[N];

    void f_sort() {
        for (int i = 1; i <= m; i++) b[i] = 0;
        for (int i = 1; i <= n; i++) b[rk[i]]++;
        for (int i = 1; i <= m; i++) b[i] += b[i - 1];
        for (int i = n; i >= 1; i--) sa[b[rk[tp[i]]]--] = tp[i];
    }
    void g_sa() {
        for (int i = 1; i <= n; i++) rk[i] = c[i], tp[i] = i;
        f_sort();
        int p = 0;
        for (int w = 1; w <= n; w <<= 1) {
            if (p >= n) break;
            p = 0;
            for (int i = n - w + 1; i <= n; i++) tp[++p] = i;
            for (int i = 1; i <= n; i++) {
                if (sa[i] > w) tp[++p] = sa[i] - w;
            }
            f_sort(), swap(rk, tp), rk[sa[1]] = p = 1;
            for (int i = 2; i <= n; i++) {
                int sl = sa[i - 1], sr = sa[i]; 
                if (tp[sl] == tp[sr] && tp[sl + w] == tp[sr + w]) {
                    rk[sa[i]] = p;
                } else rk[sa[i]] = ++p;
            }
            m = p;
        }
    }

    int main() {
        scanf("%s", c + 1);
        n = strlen(c + 1), m = 127;
        g_sa();
        for (int i = 1; i <= n; i++) {
            printf("%d ", sa[i]);
        }
        return 0;
    }
    ```

## $\text{Height}$ 数组

这部分主要讲解 $\text{Height}$ 数组，记为 $ht[i]$。

### 最长公共前缀

记 $\text{suf}(i)$ 表示后缀 $i$。

对于字符串 $S$ 和 $T$，定义其最长公共前缀 $\text{LCP}(S,T)$ 为最大的 $k(k \leq \min\{\lvert S \rvert,\lvert T \rvert\})$，使得对于任意 $i(1 \leq i \leq k)$，有 $S_i = T_i$。

记 $\text{lcp}(i, j)$ 为后缀 $sa[i]$ 与后缀 $sa[j]$ 的最长公共前缀的长度。

容易知道 $\text{lcp}(i, j) = \text{LCP}(\text{suf}(sa[i]),\text{suf}(sa[j]))$

#### $\text{LCP Lemma}$

对任意 $1 \leq i < j < k \leq n$，有

$$
\text{lcp}(i,k) = \min\{\text{lcp}(i,j),\text{lcp}(j,k)\}
$$

???+ note "证明"
    设 $p=\min\{\text{lcp}(i,j),\text{lcp}(j,k)\}$，则有 $\text{lcp}(i,j)\geq p,\text{lcp}(i,j)\geq p$。

    设 $\text{suf}(sa[i])=u,\text{suf}(sa[j])=v,\text{suf}(sa[k])=w$。

    所以 $u$ 和 $v$ 的前 $p$ 个字符相等，$v$ 和 $w$ 的前 $p$ 个字符相等。所以 $u$ 和 $w$ 的前 $p$ 个字符相等，即 $\text{lcp}(i,k)\geq p$，设其为 $q$，则 $q \geq p$。

    假设 $q > p$，即 $q \geq p + 1$，所以 $u$ 和 $w$ 的前 $p + 1$ 个字符相等。记上述性质为性质 X。
    
    又因为 $p=\min\{\text{lcp}(i,j),\text{lcp}(j,k)\}$，所以 $u[p+1]\neq v[p+1]$ 或 $v[p+1]\neq w[p+1]$，且 $u[p+1] \leq v[p+1] \leq w[p+1]$。

    但又由性质 X，$u[p+1]=w[p+1]$，即 $u[p+1] = v[p+1] = w[p+1]$，矛盾，故 $q \leq p$。

    综上所述，有 $q \geq p$，且 $q \leq p$，则 $q = p$，所以 $\text{lcp}(i,k) = p$，即 $\text{lcp}(i,k) = \min\{\text{lcp}(i,j),\text{lcp}(j,k)\}$。

    证毕。

#### $\text{LCP Theorem}$

设 $i < j$，有

$$
\text{lcp}(i, j) = \min_{i < k \leq j}\{\text{lcp}(k-1, k)\}
$$

???+ note "证明"
    令 $j=i+t$，原命题等价于

    $$
    \text{lcp}(i, i+t) = \min_{1 < k \leq i+t}\{\text{lcp}(k-1, k)\}
    $$

    对 $t$ 使用数学归纳法，当 $t=1$ 或 $t=2$ 时显然成立。

    由 $\text{LCP Lemma}$，有

    $$
    \text{lcp}(i, i+t) = \min\{\text{lcp}(i, i+t-1),\text{lcp}(i+t-1,i+t)\}
    $$

    由归纳假设，有

    $$
    \text{lcp}(i, i+t-1) = \min_{1 < k \leq i+t-1}\{\text{lcp}(k-1, k)\}
    $$

    即

    $$
    \text{lcp}(i, i+t) = \min\{\min_{1 < k \leq i+t-1}\{\text{lcp}(k-1, k)\},\text{lcp}(i+t-1,i+t)\}
    $$

    即 

    $$
    \text{lcp}(i, i+t) = \min_{1 < k \leq i+t}\{\text{lcp}(k-1, k)\}
    $$

    证毕。

#### $\text{LCP Corollary}$

对于 $i \leq j < k$，有

$$
\text{lcp}(j, k) \geq \text{lcp}(i, k)
$$

利用 $\text{LCP Theorem}$，证明显然。

### 基本定义

有定义

$$
ht[i] = \text{lcp}(i-1,i)
$$

其中 $ht[1] = 0$。

定义数组 $H[i]$，有 $H[i] = ht[rk[i]]$，即 $ht[i] = H[sa[i]]$。

### 一个重要引理

有如下引理

$$
H[i] \geq H[i - 1] - 1
$$

???+ note "证明"
    若 $H[i] \leq 1$，引理显然成立，下面我们讨论 $H[i] > 1$ 的情况。

    首先，显然有 $\text{LCP}(\text{suf}(i+1),\text{suf}(j+1)) = \text{LCP}(\text{suf}(i),\text{suf}(j))-1$（有 $\text{LCP}(\text{suf}(i),\text{suf}(j)) \geq 1$），这相当于把两个后缀都往后移了一个字符，证明略。记上述性质为性质 X。

    记 $j=sa[rk[i-1]-1]$。显然有 $\text{suf}(j) < \text{suf}(i-1)$。
    
    根据 $H$ 数组的定义，有

    $$
    \begin{aligned}
    H[i-1] &= ht[rk[i-1]] \\
           &= \text{lcp}(rk[i-1]-1,rk[i-1]) \\
           &= \text{LCP}(\text{suf}(sa[rk[i-1]-1]),\text{suf}(sa[rk[i-1]])) \\
           &= \text{LCP}(\text{suf}(j),\text{suf}(i-1)) 
    \end{aligned}
    $$

    由性质 X，有

    $$
    \text{LCP}(\text{suf}(j+1),\text{suf}(i)) = H[i-1]-1
    $$

    易知 $rk[j+1]<rk[i]$，即 $rk[j+1] \leq rk[i] - 1$。因为后缀 $j$ 和 $i-1$ 的 $\text{LCP}$ 至少为 $1$，且有 $rk[j] < rk[i-1]$，去掉第一个字符即可。

    根据 $\text{LCP Corollary}$，有

    $$
    \begin{aligned}
    \text{lcp}(rk[i]-1,rk[i]) &\geq \text{lcp}(rk[j+1],rk[i]) \\
    &= \text{LCP}(\text{suf}(rk[j+1]),\text{suf}(rk[i])) \\
    &= H[i-1]-1
    \end{aligned}
    $$

    再根据 $H$ 数组定义，有

    $$
    H[i] = \text{lcp}(rk[i]-1,rk[i])
    $$

    即

    $$
    H[i] \geq H[i-1]-1
    $$

    证毕。

### 求法

利用上面的引理，我们可以暴力地求出 $ht$ 数组，代码如下：

```cpp
void g_hei() {
    int k = 0;
    for (int i = 1; i <= n; i++) {
        if (!rk[i]) continue;
        if (k) k--;
        while (c[i + k] == c[sa[rk[i] - 1] + k]) ++k;
        ht[rk[i]] = k;
    }
}
```

???+ note "复杂度分析"
    $k$ 代表的是 $\text{LCP}$ 的长度，显然有 $k \leq n$。

    显然，代码中 `k--` 语句最多执行 $n$ 次，那么 `++k` 语句最多执行 $2 \times n$ 次，这是因为如果多于 $2 \times n$ 次，必然有一时刻 $k$ 会大于 $n$。

    这样，总复杂度为 $O(n)$。

### 一些应用

$\text{Height}$ 数组应用十分广泛。

???+ note "子串的最长公共前缀"
    有

    $$
    \text{lcp}(sa[i], sa[j]) = \min_{i < k \leq j} \{ht[k]\}
    $$
    
    这样，我们可以将原问题转化为 RMQ 问题，容易使用 ST 表或者线段树维护。

    其实这就是 $\text{LCP Theorem}$。

??? note "不同字串的数目"
    答案为

    $$
    \frac{n(n + 1)}{2} - \sum_{i=2}^n ht[i]
    $$

    证明略。

更多 $ht$ 数组的应用我们结合例题来分析。

## 例题

### [P3809 【模板】后缀排序](https://www.luogu.com.cn/problem/P3809)

模板题。

### [P4051 [JSOI2007] 字符加密](https://www.luogu.com.cn/problem/P4051)

???+ info "题意"
    给你一个长度为 $n$ 的字符串 $S$，你可以把它排成一圈，这样可以生成 $n$ 个字符串。现在对这 $n$ 个字符串进行排序，求排序后从小到大每个字符串的末尾组成的字符串。

环形似乎不好处理，但是我们有一种经典方法，将 $S$ 拼接成 $SS$，再求后缀数组即可。

### [P2852 [USACO06DEC] Milk Patterns G](https://www.luogu.com.cn/problem/P2852)

???+ info "题意"
    给你一个字符串，求出现至少 $k$ 次的子串的最大长度。

先求后缀数组和 $ht$ 数组，这样问题转化为排序后找到连续 $k$ 个后缀，使得它们的 $\text{LCP}$ 最大。其实就是在 $ht$ 数组中，求相邻 $k-1$ 个值的最小值，最后求这些最小值的最大值，单调队列维护即可。

### [P4248 [AHOI2013] 差异](https://www.luogu.com.cn/problem/P4248)

???+ info "题意"
    给定一个长度为 $n$ 的字符串 $S$，令 $T_i$ 表示它从第 $i$ 个字符开始的后缀。求

    $$ 
    \sum_{1\leq i<j\leq n}\text{len}(T_i)+\text{len}(T_j)-2\times\text{LCP}(T_i,T_j)$$

    其中，$\text{len}(a)$ 表示字符串 $a$ 的长度，$\text{LCP}(a,b)$ 表示字符串 $a$ 和字符串 $b$ 的最长公共前缀。

原式即为

$$
\sum_{1\leq i<j\leq n} (i + j) - \sum_{1\leq i<j\leq n} \text{LCP}(\text{suf}(i),\text{suf}(j))
$$

即为

$$
\frac{n(n-1)(n+1)}{2} - \sum_{1\leq i<j\leq n} \text{LCP}(\text{suf}(i),\text{suf}(j))
$$

由 $ht$ 数组的第一个应用，题目转化为求

$$
\sum_{1\leq i<j\leq n} \min_{i < k \leq j} ht[k]
$$

考虑每个 $ht[i]$ 对答案的贡献，设其为 $f[i]$，设 $j(j<i)$ 为最大的满足 $ht[j-1]>ht[i]$ 的数，则 $ht[i]$ 对 $j$ 到 $i$ 都有贡献，则有

$$
f[i] = f[j] + (i - j) \times ht[i]
$$

使用单调栈维护这个东西即可。

??? note "代码"
    ```cpp
    #include <bits/stdc++.h>
    #define int long long 
    using namespace std;

    const int N = 1e6 + 10;
    int n, m, k;
    int sa[N], rk[N];
    int b[N], tp[N];
    int ht[N], ans, f[N];
    int stk[N], top;
    char c[N];

    void f_sort() {
        for (int i = 1; i <= m; i++) b[i] = 0;
        for (int i = 1; i <= n; i++) b[rk[i]]++;
        for (int i = 1; i <= m; i++) b[i] += b[i - 1];
        for (int i = n; i >= 1; i--) sa[b[rk[tp[i]]]--] = tp[i];
    }
    void g_sa() {
        for (int i = 1; i <= n; i++) rk[i] = c[i], tp[i] = i;
        f_sort();
        int p = 0;
        for (int w = 1; w <= n; w <<= 1) {
            if (p >= n) break;
            p = 0;
            for (int i = n - w + 1; i <= n; i++) tp[++p] = i;
            for (int i = 1; i <= n; i++) {
                if (sa[i] > w) tp[++p] = sa[i] - w;
            }
            f_sort(), swap(rk, tp), p = rk[sa[1]] = 1;
            for (int i = 2; i <= n; i++) {
                int sl = sa[i - 1], sr = sa[i];
                if (tp[sl] == tp[sr] && tp[sl + w] == tp[sr + w]) {
                    rk[sa[i]] = p;
                } else rk[sa[i]] = ++p;
            }
            m = p;
        }
    }
    void g_hei() {
        int k = 0;
        for (int i = 1; i <= n; i++) {
            if (!rk[i]) continue;
            if (k) k--;
            while (c[i + k] == c[sa[rk[i] - 1] + k]) ++k;
            ht[rk[i]] = k;
        }
    }

    signed main() {
        scanf("%s", c + 1);
        n = strlen(c + 1), m = 127;
        g_sa(), g_hei();
        ans += (n * (n + 1) / 2) * (n - 1);
        for (int i = 1; i <= n; i++) {
            while (top && ht[stk[top]] > ht[i]) top--;
            int j = stk[top];
            f[i] = f[j] + (i - j) * ht[i];
            ans -= 2 * f[i];
            stk[++top] = i;
        }
        cout << ans;
        return 0;   
    }
    ```


## 参考资料

1. [后缀数组简介 - OI Wiki](https://oi-wiki.org/string/sa/)

2. [国家集训队2004论文-许智磊-后缀数组](https://github.com/OI-wiki/libs/blob/master/%E9%9B%86%E8%AE%AD%E9%98%9F%E5%8E%86%E5%B9%B4%E8%AE%BA%E6%96%87/%E5%9B%BD%E5%AE%B6%E9%9B%86%E8%AE%AD%E9%98%9F2004%E8%AE%BA%E6%96%87%E9%9B%86/%E8%AE%B8%E6%99%BA%E7%A3%8A--%E5%90%8E%E7%BC%80%E6%95%B0%E7%BB%84.pdf)

3. [字符串基础 - qAlex_Weiq - 博客园](https://www.cnblogs.com/alex-wei/p/Basic_String_Theory.html)

4. [后缀数组 最详细讲解 - ~victorique~ - 博客园](https://www.cnblogs.com/victorique/p/8480093.html)