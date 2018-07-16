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
    首先什么是采样，就是给定一个概率分布，想要获得符合这些分布的样本，这和计算概率不同，计算概率是指y=p(x)的简单计算过程，而采样是为了获得x\~p(x)。举例来说，如果给定一个高斯分布，如何获得5000个服从这个分布的数据点呢，这就是采样问题。采样的前提是给定一个x，可以计算出p(x)。<br>
    那么采样的难点在哪里呢，一般来说，计算机可以生成伪随机数，那么将之归一化到0-1区间即产生了0\~1之间的服从均匀分布的数据点。但是像高斯分布这样的如何进行采样呢，计算机并不能自己生成符合高斯分布的数据点，但是可以引入一些方法利用计算机可以产生的均匀分布来进行模拟产生其它的分布。 <br>
  * Inverse CDF <br>
    最直观的是对于一个概率密度函数p(x)，求其概率累积函数CDF，记作F(x)，那么y=F(x)是0\~1之间的值。这里出现了0\~1，自然而然想对y进行采样，这是计算机可以做到的，即产生一个服从0\~1均匀分布的y0，求与之对应的x=InvF(y0)，InvF是F的反函数（因为F单调增，所以反函数一定存在）,那么即可以进行采样了。这就是Inverse CDF Sampling的主要思想。以指数分布p(x)=lam * exp(-lam * x)为例，对其进行求概率累积函数得到：<br>
    <div align=center>
    y = F(x) = 1 - exp(-lam * x) if x > 0; 0 if x <= 0 <br>
    x = -ln(1 - y)/lam <br>
    </div><br>
    通过上面式子进行采样得到的结果如下图，左边是利用numpy里面的工具包实现的指数分布采样，右边是根据上述公式实现的采样，lam = 1.0。<br>
    <div align=center>
    <img src = "https://github.com/lxcnju/sampling/blob/master/pics/inverse_cdf.png"/>
    </div><br>
  * Box-Muller <br>
    上面介绍的对概率累积函数求逆变换的过程只适用于简单的分布，即那些可以求出来概率累积函数并且可以求出其逆变换的函数。然而大多数情况下，求概率累积函数需要进行复杂的积分操作，是不可积分的，比如高斯分布。那么如何对高斯分布进行采样呢，这里引入了对高斯分布进行操作的小技巧。引入两个高斯分布，分别是关于X和Y的高斯分布p(X)和p(Y)，形成p(X,Y)的联合概率分布，然后利用极坐标变换x=rho * sin(theta),y = rho * cos(theta)在极坐标系空间进行求积分，分别得到变量rho和theta的积分结果，即关于rho和theta的概率累积函数，然后对rho和theta进行反变换，再代入到极坐标变换公式即可。这里省略去详细过程，只列出最终结果。<br>
    <div align=center>
    U1 ~ Uniform(0, 1) <br>
    U2 ~ Uniform(0, 1) <br>
    x = sqrt(-2 * ln(U1)) * sin(2 * pi * U2) <br>
    y = sqrt(-2 * ln(U1)) * cos(2 * pi * U2) <br>
    </div><br>
    通过上面式子进行采样一维高斯，左边是利用numpy里面的工具包实现的高斯分布采样，右边是根据上述公式实现的采样。<br>
    <div align=center>
    <img src = "https://github.com/lxcnju/sampling/blob/master/pics/gau_dim1.png"/>
    </div><br>
    同样采集二维高斯如下。<br>
    <div align=center>
    <img src = "https://github.com/lxcnju/sampling/blob/master/pics/gau_dim2.png"/>
    </div><br>
  * Rejection-Sampling <br>
    拒绝采样的思想主要是为了采样p(x)，先找一个容易采样的q(x)，比如高斯分布。使得p(x) <= M * q(x)，M是个大于1的常数。这里直观理解即是使得M * q(x)将p(x)完全“包住”。那么可以很容易对q(x)进行采样得到一个点x0，那么这个点x0是否符合p(x)分布呢，这里引入拒绝采样的核心思想：生成一个0\~1之间均匀分布的随机数u，如果u <= p(x0)/(M * q(x0))，那么就接受该点，否则重新采样。这种方法很直观，可以这么简单理解（准确地来说下面说法不严格，q(x0)并不代表x0的概率）：根据q(x)分布采样到x0的概率是q(x0)， 接受它的概率是p(x0)/(M * q(x0))，那么采样到x0并接受的概率是二者相乘即p(x0)/M，则采样的数据点分布是p(x0)，因为M仅仅是一个预先选择的常数。所以拒绝采样的一个缺点就是M不容易确定，M太大，则导致接受率很低，浪费了大量计算，使得采样效率变低，M太小，某些部分不满足p(x) <= M * q(x)的条件；另外，q(x)要尽可能和p(x)像，才能保证较高的采样效率。<br>
    下面对混合高斯分布p(x) = 0.5 * N(0.0, 1.0) + 0.27 * N(-3.0, 2.0) + 0.23 * N(3.0, 0.5)进行采样，使用的q(x) = N(0.0, 10.0)，选择M = 2，得到下图。从图中可以看出，左边是p(x)和M * q(x)曲线，右边是使用拒绝采样采集的数据点的直方图，可以看出分布基本相似，即拒绝采样是有效的。实际采样5000个点，接受率为0.4986，即接受了约2500个数据点，计算采样出来的数据点的平均值为-0.081。<br>
    <div align=center>
    <img src = "https://github.com/lxcnju/sampling/blob/master/pics/rejection.png"/>
    </div><br>
  * Importance-Sampling
    基于重要性的采样和拒绝采样类似，其实现也是利用q(x)来对p(x)进行采样，这里不拒绝掉任何一个样本x，而是对每个样本x加上一个权重，权值大小w = p(x)/q(x)。这样可以利用带权重的数据点来求出一些统计量，比如均值等等，这里采样5000个数据点，得到的均值为-0.05。<br>
  * M-H
    M-H算法是基于Markov链达到平稳时即服从p(x)分布的原理进行采样。M-H算法的思想很简单，先初始化一个数据点x1，基于一个Markov链，利用转移概率分布Q(x2|x1)进行采样，然后以一定的概率接受x2，否则保持x1不变，进行下次采样。<br>
    这里涉及到Markov链的细致平稳条件，即p(x1)Q(x2|x1) = p(x2)Q(x1|x2)，那么选择的接受概率a = min(1, (p(x2)Q(x1|x2))/(p(x1)Q(x2|x1)))，随机生成一个01均匀分布的随机数u，如果u<a则接受x2。可以证明，最后采样的数据点都会服从p(x)分布。 <br>
    下面给出使用M-H算法对上面提及的混合高斯分布进行采样的结果。<br>
    <div align=center>
    <img src = "https://github.com/lxcnju/sampling/blob/master/pics/mh.png"/>
    </div><br>
  
  
      

