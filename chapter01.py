#计算损失函数，计算当前所有点的平均损失，相当于一个评价函数的东西
#最终是需要达到loss = (y - (wx + b)) ** 2 ,由于loss函数永远是大于0的函数，所以就是需要求loss的平均最小值
import numpy as np

#计算损失函数，计算当前所有点的平均损失，相当于一个评价函数的东西
#最终是需要达到loss = (y - (wx + b)) ** 2 ,由于loss函数永远是大于0的函数，所以就是需要求loss的平均最小值
def compute_error_for_line_given_points(b,w,points):
    totalError = 0;
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))

# 计算梯度信息，计算最小的值在更新new_x的时候需要使用减号，因为在需要取得最小值，所求的导数是new_x = x - learningRate * (对x求导后的结果)
#learningRate是用于控制点的速度，一般取0.001
def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points)) #获取数组行的个数
    #这里的步骤是想根据梯度下降的原则，计算损失函数的最小值，也就是loss = （W * x + b - y ）**2,在w和b的维度上分别进行损失函数的求导，
    # 并通过对所有的数据进行计算
    #相加求和平均得到平均的损失函数的导数值，再用现在的w减去乘以learningRate的w_current,从而可以得到迭代一次的新的w，b也是同理
    #实际上来说，对于模拟多个点的函数的导数，需要求相应的平均值，总体上的思路基本保持不变，也就是说需要对损失函数进行最大值或者最小值进行求导
    #此处模拟的只有两个变量控制的前提下，在有多个变量控制的前提下，对于多个损失函数的的情况也需要根据这一个原则
    #原则可以总结一下：
    # 1.求实际问题的损失函数，让损失函数最高或者最低，最高的时候采用变量s ，例如采用new_s = s_current + (learningRate * loss函数关于控制变量s_current的导数）)
    #   在求得最低点的时候采用new_s = s_current - (learningRate  * （loss函数关于控制变量s_current的导数））
    # 2.假如出现多组数据进行函数拟合的时候，需要对控制的变量进行取平均的操作，根据不同的数据带入当前控制变量的损失函数，lossFunction,分别进行对当前控制变量的求导
    #   得到当前函数的导数值，在根据不同的数据得到的导数的值进行取平均的步骤就可以得到最后的lossFunction关于s_current的导数的值。
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((w_current * x ) + b_current) )
        w_gradient += -(2/N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b,new_w]

#其中num_iterations表示的是迭代的次数
#point表示点的集合
def gradient_descent_runner(points, starting_b,starting_w,learningRate,num_iterations):
    b = starting_b #初始的b
    w = starting_w #初始的w
    for i in range(num_iterations): #迭代num_iterations次
        b,w = step_gradient(b,w,np.array(points),learningRate)
    return [b,w]

#用于运行
def run():
    #读取数据
    points = np.genfromtxt("data.csv",delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 #初始化b的值
    initial_w = 0 #初始化w的值
    num_iterations = 1000 #迭代的次数
    print("Starting gradient descent at b = {0}, w = {1},error = {2}".format(initial_b,initial_w,compute_error_for_line_given_points(initial_b,initial_w,points)))
    print("Running....")
    [b,w] = gradient_descent_runner(points,initial_b,initial_w,learning_rate,num_iterations)
    print("After {0} iterations b = {1},w = {2},error = {3}".format(num_iterations,b,w,compute_error_for_line_given_points(b,w,points)))
#程序的借口
if __name__ == '__main__':
    run()

