# 深度学习与应用

[![license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/sbl-sdsc/mmtf-spark/blob/master/LICENSE)   ![](https://img.shields.io/badge/python-3.7-brightgreen?logo=python) ![](https://img.shields.io/badge/tensorflow-2.0-brightengreen?logo=tensorflow) ![process](https://img.shields.io/badge/process-building-yellow)

| 目录                                                         | Jupyter NoteBook                                             | HTML                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [神经网络](https://nbviewer.jupyter.org/github/LibertyDream/deep_learning/blob/master/DL/neural_network.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LibertyDream/deep_learning/master?filepath=DL%2Fneural_network.ipynb) | [HTML](https://libertydream.github.io/deep_learning/DL/neural_network.html) |
| [中、英文本基础处理](https://nbviewer.jupyter.org/github/LibertyDream/deep_learning/blob/master/DL/en_zh_base_processing.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LibertyDream/deep_learning/master?filepath=DL%2Fen_zh_base_processing.ipynb) | [HTML](https://libertydream.github.io/deep_learning/DL/en_zh_base_processing.html) |
| [词向量的构建与表示](https://nbviewer.jupyter.org/github/LibertyDream/deep_learning/blob/master/DL/word_vectorizer.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LibertyDream/deep_learning/master?filepath=DL%2word_vectorizer.ipynb) | [HTML](https://libertydream.github.io/deep_learning/DL/word_vectorizer.html) |

## NLP

- embedding

[TextRank: Bringing Order into Texts](https://www.aclweb.org/anthology/W04-3252.pdf)

Word2Vec [[1]](https://arxiv.org/pdf/1301.3781)[[2]](https://arxiv.org/pdf/1310.4546)[[3]](https://arxiv.org/pdf/1402.3722v1)[[4]](https://arxiv.org/pdf/1411.2738v4)

[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)

[FastText: Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606)

[FastText: Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759)

- embedding-apply

[Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991)

[Item2Vec: Neural Item Embedding for Collaborative Filtering](https://arxiv.org/vc/arxiv/papers/1603/1603.04259v2.pdf)

[Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/abs/1803.02349)

[Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://astro.temple.edu/~tua95067/kdd2018.pdf)

- Transformer

[ELMO: Deep contextualized word representations](https://arxiv.org/pdf/1802.05365)

[Neural Machine Translation by  Jonitly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473)

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)

[GPT-2: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## 推荐系统

### 召回

- 模型召回

[The Learning Behind Gmail Priority Inbox](https://research.google.com/pubs/archive/36955.pdf)

[Amazon.com recommendations: item-to-item collaborative filtering](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)

[BPR- Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618)

[Matrix Factorization Techniques For Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

[FM: Factorization Machine](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

[FFM: Field-aware Factorization Machine](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

[Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792)

[DNN：Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://dl.acm.org/doi/10.1145/3298689.3346996)

- 行为序列召回

[GRU：Recurrent Neural Networks with Top-k Gains for Session-based Recommendations](https://arxiv.org/pdf/1706.03847)

[CNN：Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding](http://www.sfu.ca/~jiaxit/resources/wsdm18caser.pdf)

[Transformer: Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781)

- 用户兴趣拆分

召回：[Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030)

排序：[Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09248)

- 知识图谱融合

[KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/pdf/1905.07854)

[RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems](https://arxiv.org/pdf/1803.03467)

- [x] 图神经网络召回

[GraphSAGE: Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216)

[PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01973)

### 排序

- 显示特征组合

[DCN: Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf)

[xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)

[PNN: Product-based Neural Network](https://arxiv.org/pdf/1611.00144.pdf)

- [x] 特征抽取

[AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/pdf/1810.11921)

[DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction](https://arxiv.org/pdf/1804.04950)

- [x] AutoML

[FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433)

- 强化学习

[Youtube: Top-K Off-Policy Correction for a REINFORCE Recommender System](https://arxiv.org/pdf/1812.02353)

[Youtube: Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology](https://arxiv.org/pdf/1905.12767)

- [x] 多目标学习

[MMOE：Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007?download=true)

帕累托最优：[A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation](http://yongfeng.me/attach/lin-recsys2019.pdf)

- [x] 多模态融合

召回：[Collaborative Multi-modal deep learning for the personalized product retrieval in Facebook Marketplace](https://arxiv.org/pdf/1805.12312)

排序：[Image Matters: Visually modeling user behaviors using Advanced Model Server](https://arxiv.org/pdf/1711.06505)

- 长短期兴趣分离

[Neural News Recommendation with Long- and Short-term User Representations](https://www.aclweb.org/anthology/P19-1033.pdf)

[Sequence-Aware Recommendation with Long-Term and Short-Term Attention Memory Networks](http://boyuan.global-optimization.com/Mypaper/MDM-2019.pdf)

### 重排序

[Personalized Re-ranking for Recommendation](https://arxiv.org/pdf/1904.06813)

[Learning a Deep Listwise Context Model for Ranking Refinement](https://arxiv.org/pdf/1804.05936)

### 在线学习

[FTRL: Adaptive Bound Optimization for Online Convex Optimization](https://research.google.com/pubs/archive/36483.pdf)



