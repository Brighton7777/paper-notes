## OPERA : over-trust penalty and retrospection-allocation

### Attention 和 Cross-Attention

最简单的 **Self-Attention**中，我们从$Q$，$K$，$V$谈起。
对于input tokens $X \in \mathbb{R}^{N\times d_{model}}$。
我们首先会通过三个权重矩阵计算得到相应的$Q$，$K$，$V$：
$$
Q = XW_Q \\
K = XW_K \\
V = XW_V
$$
其中$W_Q\in\mathbb{R}^{d_{model}\times{d_k}}$，$W_K\in\mathbb{R}^{d_{model}\times{d_k}}$，$W_V\in\mathbb{R}^{d_{model}\times{d_v}}$，通常为实现方便令$d_k=d_v$。
之后得到Attention矩阵：
$$
Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
其中$QK^T \in \mathbb{R}^{N\times N}$由于$Q$和$K$都来自$X$，因此可以表示tokens之间的相关性。
将计算公式做如下修改，便得到了**Cross-Attention**：
$$
Q = X_tW_Q \\
K = X_eW_K \\
V = X_eW_V
$$
其中$X_t$表示decoder hidden states，即目前已经生成的tokens，$X_e$表示encoder output。
而在**跨模态的Cross-Attention**中，通过文本tokens$X_t$得到$Q$，视觉tokens$X_v$得到$K$和$V$，同时一般只更新$X_t$，而$X_v$在整个模型中保持不变，或通过**Self-Attention**更新。

### Columnar Attention Pattern

这篇文章从一个在`self-attention`反复出现的奇怪现象——**柱状注意力模式**开始分析，对$A = \text{softmax}(\frac{Q\times K^T}{\sqrt{d_k}})$可视化，如下图所示。
<p align='center'><img src="./images/columnar-attention-pattern.png" width=100%></p>
这种现象往往发生在缺少信息的token，如句号、引号等，先前的信息在这些token处发生聚合，即其他token对于该token的attention始终较强，但单一的token难以承载之前多个token的信息密度与丰富程度，绝大部分幻觉会发生在这一聚合模式出现后的10个token内。

<p align='center'><img src="./images/pattern-and-hallucination.png" width=75%></p>

这种聚合模式在LLM中十分常见。一种假设是，发生这个现象的token，起到的是总结之前的tokens，并指导后续tokens生成的作用，**summary token**。这和NLP领域中观测到的一致，即LLM在浅层中，会在一些 `anchor token` 总结先前的信息，并在深层中依靠这些 `anchor token` 生成下一个token。

这种聚合模式的出现和MLLM幻觉有强相关性。
<p align='center'><img src="./images/split-number.png" width=85%></p>

### Greedy 和 Beam Search
从数学形式看来，生成序列实际上就是**寻找概率最大的序列**，形式化地表示如下：
$$
\arg \max_y P(y | x)
$$
其中$P = \prod_{t = 1}^{T} P(y_t)$，为避免下溢常使用$\log P = \sum_{t = 1}^{T} P(y_t)$ 。
贪心的方法就是在生成每个token时，总选择概率最大的token，但局部最优往往不是全局最有。
但遍历整个空间的代价是不可接受的，因此，在效果和计算量的权衡中，就有了**Beam Search 束搜索**方法。
Beam Search中需要一个**beam size 束宽**，即最多跟踪beam size个候选项，每次将所有候选项进行生成，再保留beam search个新的候选项。
其实考虑Beam search的过程，会发现这是一个概率不断相乘，来筛选较优选项的方法，那么更长和更短的句子中，更短的句子概率往往更大，所以原始的beam search会**更偏向短句**。
往往会使用 Length Normalization 的方法，适当地平衡长短句，具体如下：

$$
\begin{aligned}
score = \frac{\log P}{T ^ \alpha}
\end{aligned}
$$

### Over-trust Penalty and Retrospecion-Allocation
在decode过程中，对每个候选项都进行评估，对over-trust的候选项进行惩罚，降低over-trust被选择的概率，即**over-trust penalty**。但可能在当前token生成之前，已经出现聚合模式，因此要支持对self-attention进行“回滚”，重新回到summary token选择其他候选项，即**retrospection allocation**。

在常见的 MLLM 中，往往通过**visual decoder**将视觉特征转化为视觉tokens，进一步通过**跨模态的映射模型**，映射到 LLM 的input space中，之后和文本tokens的embedding vector一起作为输入。形式化的有：

$$
\begin{aligned}
&\text{x}^v = \{x_0, x_1, \ldots, x_{N-1}\} \\
&\text{x}^p = \{x_N, x_{N+1}, \ldots, x_{N+M-1}\}
\end{aligned}
$$
最终的input sequence是$\{x_i\}_{t=0}^{T-1}$
在forward过程中，有：
$$
\begin{aligned}
&\textbf{h}=MLLM(\textbf{x}) \\
&\textbf{h}=\{h_0, h_1, \ldots, h_{T-1}\}
\end{aligned}
$$
所以下一个token的预测就是：
$$
p(x_t|x_{<t})=\text{softmax}(\mathcal{H}(h_t))_{x_t}
$$
最终通过不同的解码策略，选择生成的token，并加入input text的末尾，进行下一轮生成。