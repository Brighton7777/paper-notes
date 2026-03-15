

### Uncertainty 与幻觉的关系
在一次预测中，模型往往对于如问题中提到的token、功能性的token等，这些容易预测的tokens有着十足的把握，从`middle layers`开始就保持相对稳定的分布，但对于难以确定的tokens则会有着较高的uncertainty，同时伴随着最后几层分布的变动，这tokens也更容易出现幻觉。
<p align='center'><img src="./images/uncertainty.png" width=100%></p>

### “amnesia” 失忆症
文章指出，从`shallow layers`到`deep layers`的过程中，`attention`会逐步地偏向文本tokens，则视觉tokens会更难影响结果。这一问题也被`attention intervention strategies`的效果所证明确实存在。
针对这一问题，除了直接操控`attention`的方式，也可以使用`visual tokens`增强的方式，在`uncertainty`较高时，**`refresh visual memory`** 可以减缓幻觉，同时作者也对`replenish test / image`分析，发现只有`replenish image`表现最好，这也再次说明了问题切实存在且方法有效。
<p align='center'><img src="./images/replenish-test-image.png" width=100%></p>

### FFN —— “key-value memory”
最常见的`feed forward network`，由两个全连接层和中间的激活函数构成，可表示如下：
$$ FFN(x) = \phi (x W_1) {W_2^T}$$
其中$ x \in \mathbb{R}^d$，$ W_1, W_2 \in \mathbb{R}^{d\times D}$，$D$通常取$4d$。
可以将上面的`FFN`，改写成`key-value`的形式：
$$FFN(x) = \sum_{i = 1}^{D} \phi( \langle x, k_i \rangle)v_i$$
上式中的求和实际上表示按顺序组成一个向量，作为`key-value`的$k_i$和 $ v_i$ 实际上如下图所示：
<p align='center'><img src="./images/ffn-kv-memory.jpg" width=60%></p>

### Visual Retracing 视觉追溯

类似于FFN从其 `key-value memory` 进行检索，可以对视觉依据进行重新检索，即将视觉tokens作为key和value，来提供视觉有关信息。这就是**visual retracing**方法，具体可以形式化地表示如下：
$$
VR(z_v | x) = \sum_{i=1}^{N_z} \phi(\langle x, z_{v,i} \rangle)z_{v,i}
$$
其中$z_v = (z_{v,1}, \ldots, z_{v,N_v}) \in \mathbb{R} ^ {d \times N_z}$表示视觉tokens。
最终，和原始FFN整合：
$$
FFN^{(l)}(x \propto z_v)
= \alpha VR(z_v | x) + (1 - \alpha) FFN^{(l)}(x),
$$
这种使用`key-value`的检索方式，相较使用`cross-attention layer`方式开销要更小。  