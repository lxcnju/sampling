# sampling
Some methods to sampling data points from a given distribution.

从一个给定的分布里面采样，包括：Inverse CDF; Box-Muller; Rejection-Sampling; Importance-Sampling; M-H; Gibbs等方法。

* 代码架构
* 原理解析
  * 采样
  * Inverse CDF
  * Box-Muller
  * Rejection-Sampling
  * Importance-Sampling
  * M-H
  
## 代码架构
 * inverse_cdf_exp.py  以指数分布为例对累积概率函数逆变换进行采样
 * box_muller_gau.py   对高斯分布进行采样的经典算法Box-Muller实现
 * rejection_sampling.py  拒绝采样实现，以高斯分布为参照进行采样混合高斯分布
 * importance_sampling.py  重要性采样，求分布的均值等等
 * mh_sampling.py         M-H算法采样，以高斯分布为转移概率分布函数，采样混合高斯分布
 
## 原理解析
  * 采样 <br>
    首先什么是采样，就是给定一个概率分布，想要获得符合这些分布的样本，这和计算概率不同，计算概率是指y=p(x)的简单计算过程，而采样是为了获得x~p(x)。举例来说，如果给定一个高斯分布，如何获得5000个服从这个分布的数据点呢，这就是采样问题。采样的前提是给定一个x，可以计算出p(x)。<br>
    那么采样的难点在哪里呢，一般来说，计算机可以生成伪随机数，那么将之归一化到0-1区间即产生了0~1之间的服从均匀分布的数据点。但是像高斯分布这样的如何进行采样呢，计算机并不能自己生成符合高斯分布的数据点，但是可以引入一些方法利用计算机可以产生的均匀分布来进行模拟产生其它的分布。 <br>
  * Inverse CDF <br>
    最直观的是对于一个概率密度函数p(x)，求其概率累积函数CDF，记作F(x)，那么y=F(x)是0~1之间的值。这里出现了0~1，自然而然想对y进行采样，这是计算机可以做到的，即产生一个服从0~1均匀分布的y0，求与之对应的x=InvF(y0)，InvF是F的反函数（因为F单调增，所以反函数一定存在）,那么即可以进行采样了。这就是Inverse CDF Sampling的主要思想。以指数分布p(x)=lam * exp(-lam * x)为例，对其进行求概率累积函数得到：<br>
    <div align=center>
    y = F(x) = 1 - exp(-lam * x) if x > 0; 0 if x <= 0 <br>
    x = -ln(1 - y)/lam <br>
    <div><br>
    通过上面式子进行采样得到的结果如下图：<br>
    <div align=center>
    <img src = "https://github.com/lxcnju/sampling/blob/master/pics/inverse_cdf.png"/>
    <div><br>
    
  
  * Box-Muller
  * Rejection-Sampling
  * Importance-Sampling
  * M-H
  
  
      

