import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2

#绘制在同一个figure中
plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,color='red',linewidth = 2.0,linestyle = '--')#指定颜色,线宽和线型

#截取x,y的某一部分
plt.xlim((-1,2))
plt.ylim((-2,3))
#设置x,y的坐标描述标签
plt.xlabel("I am x")
plt.ylabel("I am y")
#设置x刻度的间隔
new_ticks = np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.5, 0, 1.5, 3],
           [r'$Really\ bad\ \alpha$', r'$bad$',
            r'$normal$', r'$good$', r'$very\ good$'])
           #r表示正则化,$$表示用数学字体输出
# gca = 'get current axis'
ax = plt.gca()#获取当前坐标的位置
#去掉坐标图的上和右 spine翻译成脊梁
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
#指定坐标的位置
ax.xaxis.set_ticks_position('bottom') # 设置bottom为x轴
ax.yaxis.set_ticks_position('left') # 设置left为x轴
ax.spines['bottom'].set_position(('data',0))#这个位置的括号要注意
ax.spines['left'].set_position(('data',0))
plt.show()

