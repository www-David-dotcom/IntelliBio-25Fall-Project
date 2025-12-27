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
  
## Background 基因（gene）、cDNA、CDS和蛋白质（protein）的关系
- 基因是人细胞核内染色体和线粒体DNA中可以被转录出RNA的片段及其上下游的调控序列（如用于启动的启动子、终止序列和增强子等调控序列）。基因转录出的RNA有很多种，可翻译出蛋白质的叫pre-mRNA, 不可翻译而起到调控或催化作用的统称为nc-RNA（non-coding RNA，有的数据库里这个还分好几类）。Pre-mRNA经过加工（添加头尾，促进从细胞核转移到细胞质）和剪接（pre-mRNA上会被切掉的叫内含子intron，留下接起来叫外显子exon。同一个pre-mRNA有多种剪接方式）后形成成熟的mRNA。将mRNA直接逆转录成DNA叫做cDNA。mRNA的两端是非翻译区，中间是翻译区，翻译区按照密码子表翻译成蛋白质（注意3个碱基为一个密码子，密码子之间不重叠），翻译区的序列叫做CDS。由于密码子具有简并性（即多种密码子对应同一氨基酸），根据CDS可确定蛋白质氨基酸序列，但给出氨基酸序列不能确定CDS。

## (1)	mart_export：基因、蛋白、疾病与表型的关联
- 注释：Gene stable ID, Transcript stable ID, Protein stable ID分别是通用的基因、cDNA/CDS和蛋白数据库主码，这个数据库把他们关联起来。Gene Synonym和Gene name是生物实验和医学中常用的名称，比较简短。基因的位置信息包括染色体名称、基因在该染色体上的起始和终止，以及基因所在的链（1或-1，DNA双螺旋有两条链，其中一条链编码mRNA，不同基因的编码链可能不同，距离近且在同一链上的基因一般具有更强的相关性，可能共同致病）。MIM来自OMIM数据库，它收录了所有孟德尔式遗传病（即基因与疾病有明显关联的疾病）。MIM morbid description和MIM morbid accession要么都有要么都是空白，是空白表明目前还没有发现与疾病的联系。我们可以把这两列有文本定义为致病，空白定义为不致病。Phenotype description是这个基因（突变）可能决定的表型，它和MIM有一些冲突，可暂时不管。

## (2)	9606.protein.links：蛋白质关联网络，关联分数为综合得分（0-999，一般400以上表示有有意义的关联）
- 结构：protein1-stableID protein2-stableID combined_score
- 示例：9606.ENSP00000000233 9606.ENSP00000003084 159
- 注释：仅有编号，无序列信息，编号前面带有9606.表示人，可以去除

## (3)	9606.protein.links.detailed：蛋白质关联网络，包括各种途径分析的得分明细（加权后为综合得分）
- 结构：protein1-stableID protein2-stableID neighborhood fusion cooccurence coexpression experimental database textmining combined_score
- 示例：9606.ENSP00000000233 9606.ENSP00000003084 0 0 0 0 173 0 174 159
- 注释：neighborhood（基因组相邻），fusion（基因融合），cooccurence（发育共现性）， coexpression（共表达），experimental（实验验证），database（数据库记录证据）， textmining（文献挖掘结果）。基因组相邻和基因融合表示了染色体上基因距离的临近，距离临近的基因可能共用启动子或具有较高的序列相似度，进而共表达或功能相近，都致病的概率较大；发育共现性是在不同物种的基因组中同时存在或同时缺失的情况，暗示相关性；实验验证可靠，但数据较少，有很多假阴性。如果我们的工作重点不在如何构建网络，则可忽略此数据库，直接用（2）。

## (4)	9606.protein.sequences：蛋白质的氨基酸序列
- 结构：protein-stableID protein sequence 
- 示例：>9606.ENSP00000269305
- EEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK
...

## (5)	Homo_sapiens.GRCh38.cdna.abinitio：所有预测的cDNA序列
- 示例和【注释】：>ENST00000641515.2【转录本序列名称transcript-stableID，小数点及其后部分表示版本，有的有有的没有，是数据管理者的更新，在一些数据库中可能省略，必要时删去】 cdna:abinitio chromosome:GRCh38:7:140453136:140453136:1 【染色体坐标】gene:ENSG00000157764.14【基因序列名称gene-stableID】 gene_biotype:protein_coding 【基因功能，主要有编码和非编码】transcript_biotype:protein_coding gene_symbol:BRAF【基因名称，在研究疾病时一般用这个名称而不是基因序列名】 description:B-Raf proto-oncogene, serine/threonine kinase [Source:HGNC Symbol;Acc:HGNC:1097]【基因功能，或许可以通过机器学习判断是否致病，若困难可以直接用基因－疾病关联数据库】
AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
TAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG【序列】

## (6)	Homo_sapiens.GRCh38.cdna.all：实验＋预测的cDNA序列
- 注释：与（４）的差集是所有实验的cDNA。实验发现的cDNA较少，但可靠，预测的很多，不一定可靠。

## (7)	Homo_sapiens.GRCh38.cds.all：实验＋预测的CDS序列
- 示例：>ENST00000641515.2 cdna chromosome:GRCh38:7:140453136:140475981:1 gene:ENSG00000157764.14 gene_biotype:protein_coding transcript_biotype:protein_coding gene_symbol:BRAF description:B-Raf proto-onthione kinase [Source:HGNC Symbol;Acc:HGNC:1097]
ATGTCTCACAAAA….[一定是ATG开头，一般UAA/UAG/UGA结尾]

## (8)	CTD_genes_pathways 
- 结构：GeneSymbol,GeneID,PathwayName,PathwayID,Organism,OrganismID
- 示例：TP53,7157,hsa04110:Cell cycle,KEGG:hsa04110,Homo sapiens,9606

## (9)	CTD_diseases_pathways
- 结构：DiseaseName,DiseaseID,PathwayName,PathwayID,InferenceGeneSymbol,InferenceScore
- 示例：Breast Neoplasms,MESH:D001943,hsa04110:Cellcycle,KEGG:hsa04110,CDKN1A,D014875

## (10)	Normal to CTD 
- 结构：Gene stable ID	Gene ID 
- 注释：用于关联(1)与(8)和(9)，如果不考虑通路就不用考虑(8)-(10) 
