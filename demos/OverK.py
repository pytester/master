# encoding: UTF-8
import ilib
# 策略的参数
N = 68; M1 = 12#@Over,D1

k_day = K(N, M1)        # 日线K指标
#ma_day = MA(N)
c_day = CLOSE()

week = new(W1)          # 引入周为周期
k_week = week.K(N, M1)  # 周线的K指标
g.state = u'开始'
g.trend = u''
g.k_week_idx = 0
#day_slope = make()
ema_day = c_day.EMA(400)
def slope(ind, N = 1):
    return (ind[-1] - ind[-1 - N]) / N

def onTick():
    if len(k_week) < N: return
    ma_day_slope = slope(ema_day)
    #day_slope.append(ma_day_slope)

    if ma_day_slope > 0.005:
        ema_day.show('R-')
        #day_slope.show('b-')
    elif ma_day_slope < -0.005:
        ema_day.show('G-')
        #day_slope.show('g-')

    if k_week[-1] > 60:
        if len(k_week) == g.k_week_idx:
            return
        over = ilib.over(k_week)
        if over < 0:# or ilib.over(k_day) < 0:
            if g.trend == u'牛市':
                k_week.show('rO')

            g.trend = u'牛市下拐'
            k_week.show('gv')
            g.k_week_idx = len(k_week)
            return -1
        elif over > 0:
            k_week.show('r^')
            g.k_week_idx = len(k_week)
            return 0.33
    elif k_week[-1] < 30:
            if k_day[-1] < 20:
                if ilib.over(k_day) > 0:
                    if g.trend == u'熊市':
                        k_week.show('gO')

                    g.trend = u'熊市上拐'
                    k_day.show('ri') # 画红色向下的水滴
                    g.state = u'长K低位'
                    return 0.33       #买10手
            elif ilib.over(k_week) > 0:
                if len(k_week) == g.k_week_idx: return
                g.trend = u'熊市上拐'
                k_week.show('ri') # 画红色向下的水滴
                g.state = u'长K低位'
                g.k_week_idx = len(k_week)
                return 0.5       #买10手
    else:
        if g.trend == u'牛市下拐':
            g.trend = u'熊市'
            k_week.show('go', msg='走熊')
        elif g.trend == u'熊市上拐':
            g.trend = u'牛市'
            k_week.show('ro', msg='熊市底')
        elif g.trend == u'熊市':
            #return
            pass

        over = ilib.over(k_day)
        if k_day[-1] < 10:
            if over > 0:  # 是否向上反转
                g.state = u'短K低位'
                k_day.show('gi') # 绿色向下水滴
                return 0.33
        elif k_day[-1] < 20:
            if over > 0:
                pos, val = ilib.max(c_day, -N * 3)
                c_day.show('go', pos)   # 在pos位置画绿色点
                if c_day[pos] * 0.7 > c_day[-1]:
                    g.state = u'短K超跌'
                    k_day.show('g^')    # 绿色向上箭头
                    return 0.33
        elif k_day[-1] < 50:
            if over > 0:
                if g.state == u'短K超跌' or g.state == u'短K低位':
                    k_day.show('go')
                    return 0.33
        if k_day[-1] > 20 and (g.state == u'短K超跌' or g.state == u'短K低位'):
            if over < 0:
                k_day.show('rv')
                return -1

def onLast():           # 最后调用
    return len(k_day) - g[-1][3], g.profit()

# 画指标.(指标1, 指标2, 窗口2), (指标3, 指标4, 窗口3)
# 第一窗口默认有收盘价, 最后一个窗口是总资产
vol = VOL()
show = (ema_day,), (k_day), (k_week, 30, 60), (vol, vol.MA(M1))
