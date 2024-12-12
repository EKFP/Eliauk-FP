## 引入

???+ note "模板"
    给定 $n$ 个三维点 $(x_i, y_i, z_i)$。$q$ 次查询，每次查询给定点 $(x_0, y_0, z_0)$，求满足 $x_0 \leq x_j, y_0 \leq y_j, z_0 \leq z_j$ 的 $j(j \neq 0)$ 的数量。

这相当于查询一个八分之一空间内点的个数。常见的做法有 CDQ 分治、树套树、KDT 等。这篇文章中主要讨论 CDQ 分治做法。

??? note "时间复杂度"
    - CDQ 分治：$(n + q)\log^2{n}$。
    - 树套树：$(n + q)\log^2{n}$。
    - KDT：$(n + q)\sqrt{n}$。

## 主要内容

我们结合一道例题来讲解 CDQ 分治的主要内容。

???+ note "[P3810](https://www.luogu.com.cn/problem/P3810)"
    给定 $n$ 个三维点 $(x_i, y_i, z_i)$。对于每个 $i$，求满足 $x_i \leq x_j, y_i \leq y_j, z_i \leq z_j$ 的 $j(j \neq i)$ 的数量。

我们已经知道二维数点的做法了，那么现在是三维数点。如果我们能想办法去掉一维，那么题目也就好做多了。

CDQ 分治给出的方法是，对第一维 $x$ 的**值域**进行分治。具体地，对于区间 $[l, r]$，把它分为 $[l, mid]$ 和 $[mid, r]$ 两个区间，其中 $mid$ 是区间 $[l, r]$ 的中点。

显然，左区间点的 $x$ 值一定小于等于右区间。这样，左区间的**所有**点都会对右区间的**询问**点产生贡献。这时，这一维已经满足偏序了，我们只需要统计剩下两维的偏序数量，这就是经典的二维数点问题。

??? note "代码"
    ```cpp
    #include <bits/stdc++.h>
    using namespace std;

    const int N = 2e5 + 10;
    int n, k, cnt[N];

    struct point{
        int x, y, z;
        int ans, w;
    }p[N], a[N];
    struct BIT{
        int t[N], mxk;
        BIT() {memset(t, 0, sizeof t);}
        int lb(int x) {return x & (-x);}
        void upd(int x, int tk) {
            for (int i = x; i <= mxk; i += lb(i)) t[i] += tk;
        }
        int qry(int x) {
            int res = 0;
            for (int i = x; i; i -= lb(i)) res += t[i];
            return res;
        }
    }tr;

    bool cmp1(point a, point b) {
        if (a.x == b.x) {
            if (a.y == b.y) return a.z < b.z;
            return a.y < b.y;
        }
        return a.x < b.x;
    }
    bool cmp2(point a, point b) {
        if (a.y == b.y) return a.z < b.z;
        return a.y < b.y;
    }
    void cdq(int l, int r) {
        if (l == r) return ;
        int M = (l + r) >> 1;
        cdq(l, M), cdq(M + 1, r);
        sort(a + l, a + M + 1, cmp2);
        sort(a + M + 1, a + r + 1, cmp2);
        int i = M + 1, j = l;
        while (i <= r) {
            while (a[j].y <= a[i].y && j <= M) {
                tr.upd(a[j].z, a[j].w);
                j++;
            }
            a[i].ans += tr.qry(a[i].z);
            i++;
        }
        for (int x = l; x < j; x++) {
            tr.upd(a[x].z, -a[x].w);
        }
    }

    int main() {
        scanf("%d%d", &n, &k);
        tr.mxk = k;
        for (int i = 1; i <= n; i++) {
            int x, y, z;
            scanf("%d%d%d", &x, &y, &z);
            p[i] = {x, y, z, 0, 0};
        }
        sort(p + 1, p + n + 1, cmp1);
        int tmp = 0, tot = 0;
        for (int i = 1; i <= n; i++) {
            tmp++;
            if (p[i].x != p[i + 1].x || p[i].y != p[i + 1].y || p[i].z != p[i + 1].z) {
                a[++tot] = p[i];
                a[tot].w = tmp;
                tmp = 0;
            }
        }
        cdq(1, tot);
        for (int i = 1; i <= tot; i++) {
            cnt[a[i].ans + a[i].w - 1] += a[i].w;
        }
        for (int i = 0; i < n; i++) {
            printf("%d\n", cnt[i]);
        }
        return 0;
    }
    ```


