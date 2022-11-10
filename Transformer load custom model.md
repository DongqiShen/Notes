# Transformer导入模型
使用``SentenceTransformer``预训练模型，并在模型输出层添加一个全连接层。通过``.save()``方法保存模型，并使用``Transformer``加载该模型。

## Custom pretrained model
首先使用以下代码片段，在模型的输出层，添加一个全连接层，将原本**768**维转换成**256**维，并保存。
```python
word_embedding_model = models.Transformer(model_dir, max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=256,
        activation_function=nn.Tanh()
) # 全连接层
model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
model.save(save_dir) # 保存模型
```

## 模型文件结构
原始模型文件夹``Transformer_model``内的文件结构如下所示：(``cd``至文件夹下并使用``tree``命令)
```
.
├── README.md
├── config.json
├── pytorch_model.bin
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```
添加全连接层后的模型文件结构如下所示:
```
.
├── 1_Pooling
│   └── config.json
├── 2_Dense
│   ├── config.json
│   └── pytorch_model.bin
├── README.md
├── config.json
├── config_sentence_transformers.json
├── modules.json
├── pytorch_model.bin
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```
其中``2_Dense``为我们添加的全连接层，其文件夹内的``pytorch_model.bin``是以字典形式保存的模型参数，分别为``linear.weight``和``linear.bias``。

## 使用Transformer加载模型
新模型的文件夹下面有两个``pytorch_model.bin``，分别代表了两个模型。外层文件夹的是预训练模型(``sentenceTransformer``)，``2_Dense``下为pytorch模型，即人为添加的全连接层。
1. 导入``sentenceTransformer``
```python
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)
```
2. 导入全连接层。使用``pytorch``自带的``load()``方法。
```python
fc_model = torch.load("Transformer_custom_model/2_Dense/pytorch_model.bin")
```
其中``fc_model``是一个``OrderedDict``，``KEY``为``linear.weight``和``linear.bias``，``VALUE``分别是其对应的模型参数。

在处理流程上，首先需要将每一个**768**维的向量，通过全连接层，转换成**256维**。其次，需要做一次``mean_pooling``，将句子中的每个词向量取平均值，作为整个句子的``embedding``。

全部代码如下所示：
```python
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

# 全连接层，将dimension从768 -> 256
def reduce_dimension(reduce_model, model_output):
    weight = reduce_model["linear.weight"]
    bias = reduce_model["linear.bias"]
    output = torch.matmul(model_output, weight.T)
    output = output + bias
    return output

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Sentences we want sentence embeddings for
sentences = ["到期不能按时还款怎么办", "剩余欠款还有多少？"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)

# 参数 map_location="cuda:0"，可设置模型加载到指定device
fc_model = torch.load("Transformer_custom_model/2_Dense/pytorch_model.bin")
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True,
                          truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# 1. 进行pooling，将多个词向量转换为一个句向量
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
# 2. 降维，将768维的句向量通过全连接层转换为256维
sentence_embeddings = reduce_dimension(fc_model, sentence_embeddings)

print("Sentence embeddings:")
print(sentence_embeddings.shape)
```