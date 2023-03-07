# Literature Notes

## Training language models to follow instructions with human feedback

### Card

- **Author**: OpenAI
- **Date**: 2022.04
- **Alias**: ``instructGPT``
- **Tag**: alignment, GPT-3, RLHF, fine-tune

### Introduction
随着语言模型规模的增大，依然会产生虚假，有毒，无用的信息。本文提出了``InstructGPT``，这是一个和人类意图对齐（aligned）的模型，通过使用人类反馈的数据进行微调，使得模型的输出更加符合人类的预期。实验表明，1.3B的``InstructGPT``其效果甚至比它大100倍的175B的``GPT-3``更加出色。当前大预言模型训练的目标是生成下一个单词，但是这一目标与根据人类的指示的输出这一要求并不一致，因此我们称模型是**misaligned**。为了解决这一问题，可以通过微调的方法*对齐*模型。具体地，首先使用人工手写的``Prompt``(问题+人工手写的答案)微调``GPT-3``，这一步称为**SFT(Supervised fune-tune)**。其次通过上述模型对同一个``Prompt``生成多个输出，使用人工进行打分排序，并将数据用于训练一个**Reward Model(RM)**。最后将**RM**作为``reward function``微调第一个模型，使用**PPO**算法最大化**reward**。简而言之是属于**RLHF**的方法。模型在对齐人类instruction的过程中会产生``alignment tax``，这会降低其在其他任务的性能。

论文实际上在验证``alignment``对模型的影响，结果证明``alignment``的结果相对而言是比较成功的。并且，相对于投资更大的模型而言，在对齐上进行投资的性价比更好，本文也成功验证了这是一个值得研究的方向。

### Data
大多数的``prompt``收集于早期试用版本用户提交的文本。第一个版本的数据来自于标注人员手写的``prompts``。主要有三类：
- **Plain**: 任意任务的文本，为了保持训练数据的多样性。
- **Few-shot**: 指令以及多个相关的结果。
- **User-based**: 根据用户提交的API，由标注人员编写。

根据以上的``Prompts``，产生三个不同的数据集用于微调。
1. **SFT 数据集**: 用来训练第一个SFT模型，数量为13k。
2. **RM 数据集**: 对上一个模型的输出进行人工排序，用于训练RM，数量为33k。
3. **PPO 数据集**: 没有人工标签，作为RLHF微调的输入，数量为31k。 

### Model
一共需要训练三个模型，如下所示：
1.  Supervised fine-tuning(SFT): 在人工标注的数据上对GPT-3预训练模型进行微调。训练会导致过拟合，不过模型效果会变好。
2.  Reward Modeling: 将步骤一中模型的最后一个``unembedding layer``去掉，输入是``prompt``和``reponse``，输出的是奖励分(scalar reward)。
3.  Reinforcement Learning: 使用PPO算法，结合步骤二种的RM，对步骤一中的SFT进行微调。
总的来说，训练的模型都是GPT-3，不过作者发现将RM模型降低为6B的小模型，能节省很多时间，并且175B的大模型训练不稳定。

### Methodology
下图是对整个建模过程的总结：
![InstuctGPT methodology](./image/Instructgpt%20methodology.png)

## Experiment
在三个数据集上进行了实验，分别是api prompt distribution, public NLP datasets和quanlitative results，并针对五个模型进行了对比，gpt3, gpt3(prompted), SFT, PPO和PPO-ptx。
下图是其实验的对比结果：
![InstructGPT experiment](./image/Instructgpt%20experiment.png)
其中PPO-ptx版本是将预训练的梯度和PPO的梯度相结合。

### Conclusion
很大程度上这是一篇实验性质的文章，作者主要在论证``alignment``技术的实际效果以及它带来的影响，并对这个技术的研究表现出一定的兴趣。对齐技术其实就是通过人工标注数据，使得模型的输出和人类预期的回答对齐。在论文中，作者通过大量的实验证明了``alignment``技术相比于增加模型的大小，性价比更高，取得的效果也更好，但同时会对模型在其他任务带来一定的性能损失(``alignment tax``)。如何降低这种损失，并提升模型效果，是一个开放性的研究领域。

### Appendix

1. [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)