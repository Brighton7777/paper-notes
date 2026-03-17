## DoLa : decoding by contrastive layers

这篇文章发表在2023年，是相对早期的工作，因此仅讨论了DoLa方法在LLM中是有效的，但从后面的工作中会发现，DoLa方法对于视觉幻觉的缓解有限。

文章中提出，幻觉的一个可能原因是 **maximum likelihood language modeling objective 最大似然语言建模目标**，即最小化数据和模型输出的**KL散度(KL divergence)**。
KL散度的数学表达如下：
$$
D_{KL}(P_{data} \| P_{\theta}) = \mathbb{E}_{x\sim P_{data}}[\log \frac{P_{data}(x)}{P_{\theta}(x)}]
$$
这一目标可能会造成模型过度追求“覆盖”，即会对和嵌入知识并不一致的句子赋予非零的可能性。从经验上来说，这也造成模型使用语言知识识别浅显的模式，而不是使用现实世界的真实知识进行识别和生成。
从模型可解释性上来说，在早期的层中含有更多低级别的信息，而到了更深层次之后，更多是语义层次的信息。BERT在深层中出现的“知识神经元”和事实知识可以通过修改特定feedforward layer而编辑，这也说明了这样说法的正确性。
那么这些编码在模型内部的知识能否被利用呢？
作者提出一种**contrastive decoding 对比解码**的方式，通过对比浅层和深层中的logits distribution，利用浅层中不成熟的输出，来使得深层中因为事实知识而概率提升的tokens “浮现”，进而选出正确的token。

在常见的语言模型中包含一个嵌入层，负责将tokens转化为向量序列；N个transformer layers堆叠，以及最后的仿射层，将hidden states转化成概率分布。形式化地表述如下：
$$
\mathbf{x} = \{x_1, x_2, \ldots, x_{t-1}\} \\
\mathbf{H_0} = \text{Embedding}(\mathbf{x}) = \{h_1^{(0)}, \ldots, h_{t-1}^{(0)}\}
$$
经过transformer layer之后的输出表示为$\mathbf{H_i}$，最终预测概率如下：
$$
p(x_t|x_{<t} = \text{softmax}(\phi(h_t^{(N)}))_{x_t}, \quad x_t \in \mathcal{X}
$$
为了方便比较最终输出和浅层的logits，我们使用**早退**方法，即不再只对最后一层进行仿射，而对中间每一层都仿射得到logits。
为什么在预训练中只对最后一层hidden states作用的仿射层，对其他层的输出也起到映射作用呢？
一种很重要的原因就是**残差连接**，这保证了每一层的hidden states会相对平缓的变化，而不是突变，因此对最后一层hidden states有效的仿射层，对之前的层同样奏效。
既然仿射层对所有transformer层有效，我们不妨求出所有层的logits分布：
$$
q_j(x_t|x_{<t}) = \text{softmax}(\phi(\mathbf{H_t^{(j)}}))_{x_t}, j \in \mathcal{J}
$$
为了更有效的进行对比，我们选择与$\mathbf{H_t^{(N)}}$ JS散度最大层作为浅比较层。
$$
M = \argmax_{j\in\mathcal{J}} D_{JS}(q_N(\cdot), q_j(\cdot)) \\
\hat{p}(x_t | x_{<t}) = \text{softmax}(\mathcal{F}(q_N(x_t), q_M(x_t)))_{x_t}
$$
其中$\mathcal{F}(\cdot, \cdot)$表示通过比较成熟层和未成熟层得到结果的函数。

JS散度的定义如下：
$$
M=\frac{1}{2}(P+Q) \\
D_{JS}​(P\|Q)=\frac{1}{2}D_{KL}​(P\|M)+\frac{1}{2}D_{KL}(Q\|M)
$$
相比与KL散度，JS散度是对称且有界的。