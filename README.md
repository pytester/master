# PyTester


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
3. 快速. 由上两条, 可快速实现您的策略构想.

---
## 界面

1. 鼠标移致顶部会出现:<br>
    运行, 保存, 打开, 批量计算, 批量结果, 后台输出, 代码视图切换.
2. 移出顶部后,消失.


## 策略
### 示例
1. 画出收盘价的均线
```python
show=MA(68)
```
２.画出收盘价与均线的差值(在第二窗口)
df = make(CLOSE() - MA(68))
show=None, df
３.再画出差值的均线.(在第二窗口)
df = make(CLOSE() - MA(68))
show=None, (df, df.MA(68))

４.以差值为数据源, 计算KDJ.注: 这里并不是以开盘,收盘价计算的KDJ.
df = make(CLOSE() - MA(68))
show=None, (df, df.MA(68)), (df.K(300, 50), df.D(300, 50, 20), df.J(300, 50, 20))
### 参数
每一个策略可以设置一些参数．每次运行时，可以使用不同的参数．
在代码行中用#@标识.
当您在页面点击运行时，会弹出参数框.
下图参数代码与参数框对应关系:

代码格式如下:
N = 68; M1 = 12#@策略名称,默认周期
周期:
    H1=> Hour, 小时
D1 => Day, 天
    W1=> Week, 周

### onTick() 
以上示例中并没涉及买卖的动作. 
当您需要买入或卖出. 可把这些操作放到onTick()里面.
每个周期，会调一次onTick()函数.

返回值:
        onTick()的返回值比较特殊, 为一个数字, 表示当前需要加仓, 或平仓的手数.
相当于本次变动的仓位．        
而直接return时, 表示不买也不卖.

def onTick():
    if len(ind) < N: return # 当指标长度太小时，不作操作
    return 10   # 加仓10手
#或者
    return -100 # 平仓100手, 若当前是持仓不足100手, 则全平．
### onLast()
若您需要在策略完成之后, 做一些事情. 可以把这部分代码放在onLast()里面.
策略计算完后，会调用一次onLast().
返回值:
onLast()的返回值在前台会有两地方显示.
１.当您通过点击”运行”, 测试单只合约时，结果会在”后台输出”显示．

２.当您点”批量计算”时，会计算很多只股票.　其返回值会显示在”批量结果”中

如:
1.1示例中, onLast()返回两个数字: g.profit()和len(k) – g[-1][3]
对应的在”批量结果”中会有两列.

其中参数[1], 参数[2]对应输入参数N和M1.

### show
若希望在策略的结果图上, 画上一些指标作为参考.

注:
绘图界面上, 系统默认会生成两个图像:
１.最前面的图像,为股票的收盘价
２.最后面的图像,为总资变化率.

格式:
show=(第一窗口的指标), (第二窗口的指标), ……
如:
N=68;M1=20
ma   = MA(N)
ema = EMA(N)
k      = K(N, M1)
#在第一幅图上画MA和EMA．第二副图画K线和50横线.
show=(ma, ema), (k, 50)

#若第一幅不画, 只在第二副图画K线和50横线.
show=None, (k, 50)


4.指标
4.1.用法
指标可看成是一维数组.
取天为周期.
如:
 ma = MA()

通过下标来取值.  ma[-1], ma[-2] 代表今天和昨天的均值.

下标:
    下标仅支持负数, 表示的意义为: 从今天开始, 往回倒数.
    ma[-1]      => 今天
    ma[-2]      => 昨天
    ma[-3]      => 前天
    ma[-4]      => 大前天
….

4.2.经典指标
4.2.1.KDJ
参数为N, M1, M2的KDJ 由几条指标线组成.
第二窗口画出KDJ:
N=68;M1=12;M2=12;
show=None,(\
K(N, M1),
        D(N, M1, M2),
        J(N, M1, M2) \
    )
4.2.2.MA
画出收盘价的移动平均线:
show=MA(64)

画出基于开盘价的移动平均线:
o= OPEN()
show=o.MA(64)
4.2.3.EMA

画出收盘价的指数移动平均线:
show= EMA(64)
画出基于成交量的指数移动平均线：
vol = VOL()
ema=vol.EMA(64)
show=None,(vol, ema)
4.2.4.布林带
布林线由: MＡ+ STD*2 和MＡ- STD*2
MA为均线, STD为标准差.
例:
#在第一窗口画出参数为Ｎ的布林线:
N=64
mid=MA(N)                   # 中线
std=STD(N)
up=make(mid+std * 2)    #上线(组合而成的指标)
down=make(mid - std * 2)        # 下线(组合而成的指标)
show=(up, mid, down),          # 注意,最后有个逗号, 这是python元组定义的语法

4.3.组合指标 - make
由多个指标用四则混合运算组合成新指标．用make组合完成.
如上面的布林带.
up=make(MA(N)+STD(N) * 2)    #上线(组合而成的指标)
up为生成的新指标
4.4.自定义指标 - make
若您希望希望自定义指标.  希望它能够做到以下的事情:
１.在前台画出来.
２.以它为数据源, 计算MA, EMA, KDJ等指标.
用法:
用make()初始化. 用append()填入数据.
例:
#计算收盘价与均线的差值
ind  = make()       #用户定义指标初始化.
ma = ind.MA(68) # 基于自定义指标为数据源, 计算均线.

c = CLOSE()
c_ma = c.MA(68)
def onTick():
    ind.append(c[-1] - c_ma[-1])    # 加入数据.
show=None, (ind, ma)

4.5.show()
如果您希望在某个指标上的某个位置画一个标记．ind.show()帮您完成．

函数用法:
ma=MA(68)
def onTick():
    …
    ma.show(‘gv’)    # 在ma的这个位置描一个绿色向下的箭头．
    …

样式:
支持３种图标:
    箭头:  => 符号: ^ (向上), v(向下). 此图表示为 g^
    圆点:  => 符号: o. 此图表示为go
    水滴:  => 符号:!(尖向下), i(尖向上). 此图表示为gi
    细线段           => 符号:-. 此图表示为 r-
粗线段:              => 符号:-. 此图表示为 R-

大小:
    若符号中有大写字母，则图标会大一号.否则正常大小．

颜色:
    每种图标支持３种颜色:
        g(绿色)
        r(红色)
        b(蓝色)
用法：
ma.show(‘gv’)  => 画绿色向下箭头
ma.show(‘gＶ’)  => 画绿色向下大箭头
ma.show(‘B-‘)   => 加粗的蓝色线段
一个比较完整的图象:
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



5.其它
5.1.g
预先写义好的全局变量g.
１.用于函数间传递数据．如:
def onTick():
        g.state = ‘高位’

下一次调用onTick()仍可读到state的值．
２.交易记录
g[-1][0]    => 上一次交易手数
g[-1][2]    => 上一次交易时的价格．
g[-1][3]    => 上一次交易时, 收盘价格指标的长度．
３.汇总
g.profit()  => 计算现在的总资产.
5.2.help
帮助信息
help(MA)    => 查看MA用法
help(make)  => 查看make用法

5.3.new()
若您需要在策略里面引入以下内容:
１.多支合约
２.多个周期
可使用如下格式:
１.引入其它合约.
symbol = new(‘000001’)  => 引入平安银行.
symbol = new(‘S000001’)     => 引入上证指数.
２.引入其它周期.
week = new(W1)      => 引入一周为周期.
day  = new(D1)      => 引入一天为周期
３.引和其它合约的不同周期
symbol = new(‘000001’, D1)      => 引入一天为周期的平安银行.
symbol = new(‘S000001’, W1)     => 引入一周为周期的上证指数.
引用之后的使用方法:
    引用之后的用法与正常指标完全相同.如:
sb = new(‘000001’, D1)
ma = sb.MA()            => 移动平均线
k   = sb.K(68, 12)      => K值
c   = sb.CLOSE()        => 收盘价
示例:
１.引入上证指数收盘价:
show=None, new('S000001').CLOSE()

２.引入上证指数一周为周期的收盘价. 从图上看起来没啥差别, 实际颗粒更粗
show=None, new('S000001', W1).CLOSE()

３.引入上证指数一周为周期的收盘价, 并画出均线. 从图上看起来没啥差别, 实际颗粒更粗
c = new('S000001', W1).CLOSE()
show=None, (c, c.MA(68))

４.同上, 仅把周期改为天.
c = new('S000001').CLOSE()
show=None, (c, c.MA(68))



6.批量计算
6.1.参数优化
当您拿不准策略的参数设定为多少, 才最优.
可以使用批量计算:
在批量计算的弹出框内输入参数的范围.


如上图, 参数N设置为50 80 3. 
系统会帮您把两个参数
N  => 50, 53, 56, .. 80    # 间隔为3
M1=> 8, 10, … 15    # 间隔为2
所有组合对应的策略, 都计算一遍.

批量结果里面会看到所有的参数, 以及对应的onLast()的返回值.


通过对返回值进行大小排序, 可得到最优的参数组.
6.2.策略选股
当您希望知道哪些股票已经超跌, 或者适合买入.
您可以通过批量计算, 选译”上证50所有股票”等选项.




它会把所有股票都计算一遍.
结果如下:


您可以通过对结果排序来实现优选.
