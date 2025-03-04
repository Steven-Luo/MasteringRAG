# 说明

本项目是一个使用LLM（大语言模型）使用RAG技术构建文档问答的项目，将会涵盖企业构建基于RAG的文档问答几乎所有的常见优化手段。
项目重点介绍算法流程，不会将重点放在非常规范化的工程代码上，因此，每一个Notebook文件都可以独立运行，不会做公共逻辑的抽象。
具体包括如下话题：

## RAG系列

- 问答数据构建
  - [使用RAG技术构建企业级文档问答系统之QA抽取](https://mp.weixin.qq.com/s?__biz=MjM5NTQ3NTg4MQ==&mid=2257496784&idx=1&sn=94a1afc05728f0c7d8cf92004125f392&chksm=a58df21692fa7b00104850fe8dfb287acb78f149df77bff7f7d23cc7d18c3998814f08924d8a&token=2031500795&lang=zh_CN#rd)
- Baseline搭建
  - [使用RAG技术构建企业级文档问答系统之基础流程](https://mp.weixin.qq.com/s/P_XWrQtOyE1gwnQ0d1Putg)
- 检索优化
  - [检索优化(1)Embedding微调](https://mp.weixin.qq.com/s/C06SXepnw49GC1UtNvpFcA)
  - [检索优化(2)Multi Query](https://mp.weixin.qq.com/s/NCsxMqkAQEGSLCxDXU_mkA)
  - [检索优化(3)RAG Fusion](https://mp.weixin.qq.com/s/T-qeEkanLs9XX0oOwdL5_g)
  - [检索优化(4)BM25和混合检索](https://mp.weixin.qq.com/s/KFrSqG6mZb0TPgbHlgZ9dA)
  - [检索优化(5)常用Rerank对比](https://mp.weixin.qq.com/s/It50F1OmYOHNOs0KRFJ0Lg)
  - [检索优化(6) Rerank模型微调](https://mp.weixin.qq.com/s/1revSlQsum5uRF9U_OYRTA)
  - [检索优化(7)HyDE](https://mp.weixin.qq.com/s/62UWBMV24RDePcGdYAZW_Q)
  - [检索优化(8)Step-Back Prompting](https://mp.weixin.qq.com/s/DxK9rUeG_4ZMvD2_oopWZg)
  - [检索优化(9)Parent Document Retriever](https://mp.weixin.qq.com/s/hq-9E_vuRhZs7Ex_TcZUbA)
  - [检索优化(10)上下文压缩](https://mp.weixin.qq.com/s/_sRv-xNuy-REWUiV3-_8CA)
  - [检索优化(11)上下文片段数调参](https://mp.weixin.qq.com/s/mEm1fdRW7igNK8bRJ-8heA)
  - [检索优化(12)RAPTOR](https://mp.weixin.qq.com/s/4zHMb2uJrTXEbHpNtz5LHg)
  - [检索优化(13)Contextual Retrieval](https://mp.weixin.qq.com/s/umrgtJ6H2WL0p7HL_gcmkw)
  - [检索优化(14)CRAG——自动判断是否联网检索的RAG](https://mp.weixin.qq.com/s/B7SqodOv0T8YlglFW58prA)
- 文档解析优化
  - [解析(1)使用MinerU将PDF转换为Markdown](https://mp.weixin.qq.com/s/E35jqTA2t_5Sh35AIopvGw)
- 文档切分优化
  - [切分(1)Markdown文档切分](https://mp.weixin.qq.com/s/epIIyv9lQDWDtZZXC4GE1w) 
  - [切分(2)使用Embedding进行语义切分](https://mp.weixin.qq.com/s/saEr5vNLw-gu9xRUwJLqsg)
  - [切分(3)使用Jina API进行语义切分](https://mp.weixin.qq.com/s/4OaOJb2uFHOjIqBLs97y6Q)
  - [切分(4)Meta Chunking](https://mp.weixin.qq.com/s/dEeKdDRvaucCWnj-PR2zfQ)
  - [切分(5)Late Chunking](https://mp.weixin.qq.com/s/PDl84dSd345wU3FUarAz0g)
- 生成优化
  - [生成优化(1)超长上下文LLM vs. RAG](https://mp.weixin.qq.com/s/n0RLhQNcWRPKNBJwaX-a2g)
- 新架构
  - [新架构(1)LightRAG](https://mp.weixin.qq.com/s/TVtCCeHEK-_raP8S05w2Rg)
- 评估
  - [评估(1)TruLens进行评估](https://mp.weixin.qq.com/s/4SNaZT8sC6LOL-K8TkHgMw)
  - [评估(2)使用GPT4进行评估](https://mp.weixin.qq.com/s/332MeDhzAns_t8dvMOgnYQ)
- 使用Flowise零代码构建RAG
  - [使用Flowise零代码构建RAG(1)——基础流程](https://mp.weixin.qq.com/s/BPKwN4feV828aFL7NbgxHw)
  - [使用Flowise零代码构建RAG(2)——HyDE](https://mp.weixin.qq.com/s/zq0Tuk5g_o5Ros1rnY2r5w)
  - [使用Flowise零代码构建RAG(3)——Reciprocal Rank Fusion](https://mp.weixin.qq.com/s/jkmikh9b4okdWbeq1kGoyw)
  
## Agent系列

- [Langchain中使用Ollama提供的Qwen大模型进行Function Call实现天气查询、网络搜索](https://mp.weixin.qq.com/s/1UKb_Iii9-Hhp-EJTjjPpQ)
- [Langchain中使用千问官方API进行Function Call实现天气查询、网络搜索](https://mp.weixin.qq.com/s/tGeX7gX0JPE7x55Po-zQIw)
- [使用Ollama提供的Llama3 8B搭建自己的斯坦福多智能体AI小镇](https://mp.weixin.qq.com/s/L9fJcicD4GlGHS89H6thrg)
- [使用Ollama提供的Qwen2 7B搭建自己的中文版斯坦福多智能体AI小镇](https://mp.weixin.qq.com/s/RHxW_2vP0Y8JS6xsTyRJnA)

# 公众号

欢迎大家关注我的公众号，关注LLM、Langchain、Agent、Knowledge Graph等话题，会定期开源一些项目。

![](assets/qrcode_for_gh_5aecbba21fec_430.jpg)
