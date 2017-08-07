# PyTester

---
## 目录
* [界面](#界面)
* [策略](#策略)
    * [示例](#示例)
    * [参数](#参数)
    * [`onTick()`](#ontick)
    * [`onLast()`](#onlast)
    * [`show()`](#show)
* [指标](#指标)
    * [用法](#用法)
    * [经典指标](#经典指标)
        * [KDJ](#kdj)
        * [MA](#ma)
        * [EMA](#ema)
        * [布林带](#布林带)
        * [组合指标](#组合指标)
        * [自定义指标](#自定义指标)
        * [`指标.show()`](#指标show)
* [其它](#其它)
    * [全局变量g](#全局变量g)
    * [`help()`](#help)
    * [`new()`](#new)
* [批量计算](#批量计算)
    * [参数优化](#参数优化)
    * [策略选股](#策略选股)

---
## 前言

PyTester是一个股票策略回测的平台.
用python语言．所以适合于有一定python基础同学．

---
### 作用

1. 快速建立策略．
2. 数据分析．
3. 股票筛选．

---
* 测试平台: www.pytester.com
* 浏览器: 对低版本IE兼容性支持还不完整, 建意使用最新的IE或firefox, chrome浏览器打开


---
### 特点

1. 简单. 指标使用傻瓜化.
2. 灵活. 感谢python带来的强大灵活性
3. 快速. 可快速实现您的策略构想.

---
## 界面

1. 鼠标移致顶部会出现:<br>
    运行, 保存, 打开, 批量计算, 批量结果, 后台输出, 代码视图切换.

![](/images/menu.jpg)

2. 移出顶部后,消失.


## 策略
### 示例
1. 画出收盘价的均线
```python
show=MA(68)
```

![](/images/show_ma_68.jpg)

2. 画出收盘价与均线的差值(在第二窗口)
```python
df = make(CLOSE() - MA(68))
show=None, df
```

![](/images/close_ma68.jpg)

3. 再画出差值的均线.(在第二窗口)
```python
df = make(CLOSE() - MA(68))
show=None, (df, df.MA(68))
```

![](/images/df_ma.jpg)

4. 以差值为数据源, 计算KDJ.
    注:
        这里并不是以开盘,收盘价计算的KDJ.

```python
df = make(CLOSE() - MA(68))
show=None, (df, df.MA(68)), (df.K(300, 50), df.D(300, 50, 20), df.J(300, 50, 20))
```

![](/images/df_kdj.jpg)

---
### 参数

>每一个策略可以设置一些参数．每次运行时，可以使用不同的参数.<br>
>在代码行尾用 #@ 标识.<br>
>当您在页面点击运行时，会弹出参数框.<br>
>下图参数代码与参数框对应关系:<br>

* 代码格式如下:
```python
N = 68; M1 = 12#@策略名称,默认周期
```

* 周期:
    * H1 -> Hour, 小时
    * D1 -> Day, 天
    * W1 -> Week, 周
* 例:<br>
    * N=68; M1=12#@策略名称,D1<br>
        运行时,弹出对话框<br>
![](/images/run_setting.jpg)

---
### onTick

以上示例中并没涉及买卖的动作.<br>
当您需要买入或卖出. 可把这些操作放到onTick()里面.<br>
每个周期，会调一次onTick()函数.<br>

* 返回值:<br>
    1. onTick()的返回值比较特殊, 为一个数字. 
    2. 表示当前需要加仓, 或平仓的手数或百分比. 相当于本次变动的仓位.
        如return x
        * 当 x 大于1时或小于-1时, 按手数买卖.
        * 当 x 在-1 ~ 1时, 按百分比买卖.
        * x 为正时, 表示买入. 为负时, 表示卖出.
    4. 而直接__return__时, 表示不操作.

* 例:<br>

```python
def onTick():
    if len(ind) < N: return # 当指标长度太小时，不作操作
    return 10   # 加仓10手
#或者
    return -100 # 平仓100手, 若当前是持仓不足100手, 则全平．
#或者
    return 0.33 # 加仓1/3仓位.
#或者
    return 1    # 全仓买入.
#或者
    return -0.33 # 平1/3仓位.
#或者
    return -1   # 全部平仓.
```

---
### onLast

若您需要在策略完成之后, 做一些事情. 可以把这部分代码放在onLast()里面.
策略计算完后，会调用一次onLast().

* 返回值:<br>
    onLast()的返回值在前台会有两地方显示.
    1. 当您通过点击”运行”, 测试单只合约时，结果会在”后台输出”显示．

    2. 当您点”批量计算”时，会计算很多只股票.　其返回值会显示在”批量结果”中

* 如:<br>
    1.1示例中, onLast()返回两个数字: g.profit()和len(k) – g[-1][3]
    对应的在”批量结果”中会有两列.

    其中参数[1], 参数[2]对应输入参数N和M1.

---
### show

若您希望网页上, 画一些指标作为参考. <br>
可使用如下:<br>
```python
show = A, B, ... 
```
来显示各种指标.

* 注:<br>
    绘图界面上, 系统默认会生成两个图像:
    1. 最前面的图像,为股票的收盘价
    2. 最后面的图像,为总资变化率.

* 格式:<br>
```python
show=(第一窗口的指标), (第二窗口的指标), ...
```
* 例:<br>
```python
N=68;M1=20
ma   = MA(N)
ema = EMA(N)
k      = K(N, M1)
#在第一幅图上画MA和EMA．第二副图画K线和50横线.
show=(ma, ema), (k, 50)

#若第一幅不画, 只在第二副图画K线和50横线.
show=None, (k, 50)
```
效果如下图<br>

* show=(ma, ema), (k, 50)<br>
    第一窗口画ma和ema, 第二窗口画k和50线, 第三窗口是总资产.
![](/images/ema_k50.jpg)

* show=None, (k, 50)
    第一窗口只有默认的收盘价, 第二窗口画k和50线, 第三窗口是总资产.
![](/images/None_k50.jpg)

## 指标
### 用法
指标可看成是一维数组.<br>

假设周期为天.<br>
* 如:
```python
 ma = MA()
```

通过下标来取值.  ma[-1], ma[-2] 代表今天和昨天的均值.

* 下标:<br>
下标仅支持负数, 表示的意义为: 从今天开始, 往回倒数.
```
    ma[-1]      -> 今天
    ma[-2]      -> 昨天
    ma[-3]      -> 前天
    ma[-4]      -> 大前天
```

### 经典指标
#### KDJ
参数为N, M1, M2的KDJ 由几条指标线组成.<br>
第二窗口画出KDJ:<br>
```python
N=68;M1=12;M2=12;
show=None,(\
K(N, M1),
        D(N, M1, M2),
        J(N, M1, M2) \
    )
```
#### MA
画出收盘价的移动平均线:
```python
show=MA(64)
```

画出基于开盘价的移动平均线:
```python
o= OPEN()
show=o.MA(64)
```

#### EMA

画出收盘价的指数移动平均线:
```python
show= EMA(64)
```

画出基于成交量的指数移动平均线：
```python
vol = VOL()
ema=vol.EMA(64)
show=None,(vol, ema)
```
#### 布林带
布林线由: MA + STD * 2 和MA - STD * 2, 组成.<br>
其中, MA为均线, STD为标准差.<br>

例:
```python
#在第一窗口画出参数为N的布林线:
N=64
mid=MA(N)                   # 中线
std=STD(N)
up=make(mid+std * 2)    #上线(组合而成的指标)
down=make(mid - std * 2)        # 下线(组合而成的指标)
show=(up, mid, down),          # 注意,最后有个逗号, 这是python元组定义的语法
```

![](/images/boll_ma.jpg)

#### 组合指标
由多个指标用四则混合运算组合成新指标．用make(A + B)的形式完成.<br>
如上面的布林带.<br>
```python
up=make(MA(N)+STD(N) * 2)    #上线(组合而成的指标)
```
up为生成的新指标

#### 自定义指标

若您希望希望自定义指标.  希望它能够做到以下的事情:<br>
    1. 在前台画出来.<br>
    2. 以它为数据源, 计算MA, EMA, KDJ等指标.

    用法:
        ind = make()    # 初始化
        ind.append(x)   # 添加数据

例:<br>
```python
#计算收盘价与均线的差值
ind  = make()         # 用户定义指标初始化.
ma   = ind.MA(68)     # 基于自定义指标为数据源, 计算均线.

c    = CLOSE()
c_ma = c.MA(68)
def onTick():
    ind.append(c[-1] - c_ma[-1])    # 加入数据.
show=None, (ind, ma)
```

![](/images/usr_df_ma.jpg)

#### 指标.show
如果您希望在某个指标上的某个位置画一些标记．ind.show()帮您完成．<br>

* 函数用法:<br>
```python
ma=MA(68)
def onTick():
    …
    ma.show('gv')    # 在ma的这个位置描一个绿色向下的箭头．
    …
```
* 样式:<br>
    支持３种图标:

    |形状|符号|示例|效果|
    |----|----|----|----|
    |箭头| ^  | g^ |![](/images/g_up_arraw.jpg)|
    |圆点| o  | go |![](/images/g_o.jpg)|
    |水滴| i  | ri |![](/images/r_i.jpg)|
    |线条| -  | r- |![](/images/r_line.jpg)|

* 方向:<br>
    箭头与水滴是带方向的. <br>

    |符号|方向|
    |----|----|
    |^   |向上箭头|
    |v   |向下箭头|
    |i   |向下水滴|
    |!   |向上水滴|

* 大小:<br>
    若符号中有**大写**字母，则图标为**大号**.否则正常大小．<br>

* 颜色:<br>
    总共支持3种颜色:

    |符号 | 颜色 |
    |--|-----|
    |g | 绿色|
    |r | 红色|
    |b | 蓝色|

用法：<br>
    ma.show('gv')   -> 画绿色向下箭头<br>
    ma.show('gV')  -> 画绿色向下大箭头<br>
    ma.show('B-')   -> 加粗的蓝色线段<br>

一个比较完整的图象:<br>
```python
from  ilib import over
k = K(200, 50)
def onTick():
    if len(k) < 200: return
    ov = over(k)
    lastk = k[-1]
    if lastk < 20 and ov > 0:    # 小于20, 并上拐时, 画红色向上的箭头
        k.show('r^')
    elif lastk > 75 and ov < 0: # 大于75, 并下拐时, 画绿色向下的箭头
        k.show('gv')
show = None, k
```

## 其它
---
### 全局变量g
全局变量g.<br>
* 函数间传递数据.
如:<br>

```python
g.state = ''
def onTick():
    g.state = '高位'
```

下一次调用onTick()仍可读到state的值．


* 查看交易记录<br>
    g本身是一个数据. 它存储了策略onTick()的返回值.

|g|数据|
|-------|----------------------|
|g[-1][0]|上一次交易手数       |
|g[-1][2]|上一次交易时的价格   |
|g[-1][3]|上一次交易时, 收盘价格指标的长度|


* 汇总

    在onLast()中, 可调用 g.profit(), 计算现在的总资产.<br>

### help()

    *帮助信息<br>
    help()      -> 输出当前支持的指标<br>
    help(MA)    -> 查看MA用法<br>
    help(make)  -> 查看make用法<br>

### new()
若您需要在策略里面引入以下内容:
* 多支合约
* 多个周期

可使用如下格式:
* 引入其它合约<br>
    * symbol = new(‘000001’)      -> 引入平安银行.<br>
    * symbol = new(‘S000001’)     -> 引入上证指数.<br>

* 引入其它周期.
    * week = new(W1)      -> 引入一周为周期.
    * day  = new(D1)      -> 引入一天为周期

* 引和其它合约的不同周期
    * symbol = new(‘000001’, D1)      -> 引入一天为周期的平安银行.
    * symbol = new(‘S000001’, W1)     -> 引入一周为周期的上证指数.

* 引用之后的使用方法:<br>
    引用之后的用法与正常指标完全相同.如:

```python
sb = new('S000001', D1)    # 引入上证指数
ma = sb.MA(68)             # 移动平均线
k   = sb.K(68, 20)         # K值
c   = sb.CLOSE()           # 收盘价
show = None, (c, ma), k
```
![](/images/sangzheng_ma_k.jpg)


* 示例:
    * 引入上证指数收盘价:
    ```python
    show=None, new('S000001').CLOSE()
    ```
    ![](/images/huaying_shangzheng.jpg)

    * 引入上证指数一周为周期的收盘价. 从图上看起来没啥差别, 实际颗粒更粗
    ```python
    show=None, new('S000001', W1).CLOSE()
    ```
    ![](/images/huaying_shangzheng_w1.jpg)

    * 引入上证指数一周为周期的收盘价, 并画出均线.
    ```python
    c = new('S000001', W1).CLOSE()
    show=None, (c, c.MA(68))
    ```
    ![](/images/huaying_shangzheng_w1_ma.jpg)

    * 同上, 仅把周期改为天.
    ```python
    c = new('S000001').CLOSE()
    show=None, (c, c.MA(68))
    ```
    ![](/images/huaying_shangzheng_d1_ma.jpg)


## 批量计算
### 参数优化
当您拿不准策略的参数设定为多少, 才最优.<br>
可以使用批量计算:<br>
在批量计算的弹出框内输入参数的范围.<br>

![](/images/search_setting.jpg)

如上图, 参数N设置为50 80 3. M1设置为8 15 2.<br>
系统会帮您把两个参数<br>
    N  -> 50, 53, 56, .. 80    # 间隔为3<br>
    M1 -> 8, 10, … 15          # 间隔为2<br>
所有组合对应的策略, 都计算一遍.<br>

批量结果里面会看到所有的参数, 以及对应的onLast()的返回值.<br>

![](/images/search_args_result.jpg)

通过对返回值进行大小排序, 可得到最优的参数组.

### 策略选股
当您希望知道哪些股票已经超跌, 或者适合买入.<br>
您可以通过批量计算, 如选译”上证50所有股票”等选项.<br>

![](/images/searh_sb_sz50.jpg)

它会把所有股票都计算一遍.<br>
结果如下:<br>

![](/images/searh_sb_sz50_result.jpg)

您可以通过对结果排序来实现优选.<br>
