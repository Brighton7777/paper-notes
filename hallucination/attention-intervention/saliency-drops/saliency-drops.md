## Hallucination Begins Where Saliency Drops

在幻觉缓解领域中，早期的方法往往缺乏幻觉的成因，导致解释性较差，且对外部知识和额外计算开销较大。现有的基于注意力的方法并不能十分可靠地区分幻觉和正确输出，作者认为这是因为其只使用前向传递的attention，而忽略了**梯度**所隐含的信息。文章提出了一种融合传统attention和梯度信息的检测工具，来检测每个token的“强度”，并且作者推断如果先前输出的tokens对于下个token的预测有着较小的**saliency 显著性**，就发生了幻觉。据此作者提出相应的缓解方法。

### Attention Intervention 方法奏效的真正原因
现有的注意力干涉方法中，认为当一个token有着较高的attention权重时，这种over-trust / over-reliance可能会导致丧失对先前token的关注，进而造成幻觉。但attention和幻觉的关系并没有被很好地解释，是因为将attention可视化为热力图之后，只能反映模型在推理过程中的选择，而没有揭示输入tokens的变化是如何影响最终输出的。作者抓住了被现有方法广泛忽视的**gradient information 梯度信息**，这对理解生成过程中的tokens之间的关系是很重要的。
在某些attention中没有出现“聚合模式”但仍然出现幻觉，作者认为要更进一步发掘幻觉产生的原因(why and where)，而不是仅仅知道什么时候发生了幻觉(when)。
为什么会发生“知识聚合”呢？
在这里要介绍一个概念——**information flow 信息流**，其被提出于2023年的文章“Label Words are Archors”中。information flow的含义是，一个token对另一个token预测的影响程度，可以使用梯度视角分析，即$\frac{\partial P(Y)}{\partial X}$，也可以从注意力的token间的“关注度”视角分析。
从梯度视角得到启发，作者同时考量attention权重的内积和梯度，定义一种非监督的指标——LVLMs-Saliency，用来衡量生成当前token时先前token的影响。
根据LVLMs-Saliency，观测到不同于传统Attention Heat Map的模式，如下图所示：
<p align='center'><img src="./images/saliency-pattern.png" width=100%></p>
