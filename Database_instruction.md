# 数据库说明
## DisGeNET
-  我们可以只关注两个带【】的指标
- 【Gene】: Gene name or Gene synonym in DB1, ie: CFTR。如果要查基因序列，需要由基因名根据dataset_1对应序列号，然后根据dataset_5对应序列。
- Disease: Disease name, ie: cystic fibrosis 囊性纤维化
- 【Score_gda】: Comprehensive score of gene-disease association，是所有GDA里最综合的分数， 0-1，越高基因与疾病的关联越强
- N PMIDs: PubMed上这种基因相关文章数量，越高一般意味着越是研究热点，证据充分。
- N variants_gda: 使该基因致病的突变数量。一个基因的多种突变都可能致病，N variants_gda/N PMIDs的比值高暗示致病机制复杂，低则暗示个别关键的突变严重影响功能。
- EI: 反映所有有关该基因的文献中，支持该基因与疾病有关的比例
- EL_gda: 另一种计算方法得到的离散的基因-疾病关联指标
- DSIg: disease specificity score，基因与疾病关联的专一程度，基因如果只关联这种疾病，分数就高，关联多种则分数低。
- DPIg: phenotype specificity score，基因与疾病特定临床表型的关联，一个疾病可能有多个临床表型，可能与不同的基因关联
- pLI: probability of loss-of-function intolerance，突变（致病）基因被自然选择排除（没法留下后代）的程度，比如囊性纤维化就很低，它是隐性基因，携带者没有症状还有一些优势，因而被自然选择的程度较低，可以流传
  
