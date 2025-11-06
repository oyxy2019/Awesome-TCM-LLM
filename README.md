# Awesome-TCM-LLM

此仓库整理了现有中医药大模型(TCM-LLM)相关资源，旨在为研究人员提供全面的参考。

欢迎通过PR参与贡献，共同促进中医药人工智能发展。

# 目录

- [Awesome-TCM-LLM](#awesome-tcm-llm)
- [目录](#目录)
- [基准数据集](#基准数据集)
    - [纯文本【TODO】](#纯文本todo)
    - [多模态](#多模态)
- [综述](#综述)
- [方法（开源研究）](#方法开源研究)
    - [ShenNong-TCM](#shennong-tcm)
    - [CMLM-ZhongJing](#cmlm-zhongjing)
    - [TCMLLM-PR](#tcmllm-pr)
    - [Huang-Di](#huang-di)
    - [MCM](#mcm)
    - [LingdanLLM](#lingdanllm)
    - [BianCang](#biancang)
    - [TCMChat](#tcmchat)
    - [JingFang](#jingfang)
    - [OpenTCM](#opentcm)
    - [TCM-KLLaMA](#tcm-kllama)
    - [Tianyi](#tianyi)
    - [DoPI](#dopi)
    - [ShizhenGPT](#shizhengpt)
    - [ZhiFangDanTai](#zhifangdantai)
    - [TianHui](#tianhui)
- [方法（闭源产品）](#方法闭源产品)
    - [岐黄问道·中医大模型](#岐黄问道中医大模型)
    - [聪宝素问](#聪宝素问)
    - [九为中药大模型](#九为中药大模型)
    - [天河灵枢大模型](#天河灵枢大模型)
    - [本草智库·中药大模型](#本草智库中药大模型)
    - [数智本草](#数智本草)
    - [海河·岐伯](#海河岐伯)
    - [数智岐黄2.0](#数智岐黄20)
    - [中医横琴大模型](#中医横琴大模型)
- [Citation](#citation)


# 基准数据集

### 纯文本【TODO】
待更新

### 多模态

| **数据集** | **机构** | **时间** | **多模态** | **Size** | **Source** | **备注** |
| --- | --- | --- | --- | --- | --- | --- |
| ZhongJing-OMNI | 复旦大学 | 2024-10 | 图像、文本 | 暂无 | ZhongJing-OMNI: The First Multimodal Benchmark for Evaluating Traditional Chinese Medicine \[[repo](https://github.com/pariskang/ZhongJing-OMNI)\] \[[data](https://huggingface.co/datasets/CMLM/ZhongJing-OMNI)\] | 暂未开源 |
| TCM-Ladder | 上海中医药大学 | 2025-05 | 图像、音频、视频、文本 | 52,169 | TCM-Ladder: A Benchmark for Multimodal Question Answering on Traditional Chinese Medicine \[[paper](https://arxiv.org/abs/2505.24063)\] \[[repo](https://github.com/orangeshushu/TCM-Ladder)\] \[[data](https://huggingface.co/datasets/timzzyus/TCM-Ladder)\] | 单诊独立评估 |
| TCM-Text-Exams (ShizhenGPT) | 港中深FreedomAl | 2025-08 | 文本 | 1088 | ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine \[[paper](https://arxiv.org/abs/2508.14706)\] \[[data](https://huggingface.co/datasets/FreedomIntelligence/TCM-Text-Exams)\] | 单诊独立评估，其他模态暂未开源 |

# 综述

1.  大语言模型在中医药领域的应用、挑战与前景 \[2024-05\] \[[paper](https://www.aminer.cn/pub/67a70e10ae8580e7ffb1a29b/application-challenges-and-prospects-of-large-language-model-in-the-field-of)\]
    
2.  Large Language Models in Traditional Chinese Medicine: A Scoping Review \[2024-12\] \[[paper](https://www.aminer.cn/pub/67582e60ae8580e7ff454b3c/large-language-models-in-traditional-chinese-medicine-a-scoping-review)\]
    
3.  Advancements in Artificial Intelligence-Driven Diagnostic Models for Traditional Chinese Medicine \[2025\] \[[paper](https://www.aminer.cn/pub/682798cb163c01c850dcff7d/advancements-in-artificial-intelligence-driven-diagnostic-models-for-traditional-chinese-medicine)\]
    
4.  Application of Large Language Models in Traditional Chinese Medicine: A State-of-the-Art Review \[2025\] \[[paper](https://www.aminer.cn/pub/68644b07163c01c85082e4cc/application-of-large-language-models-in-traditional-chinese-medicine-a-state-of)\]
    
5.  Can GPTs Accelerate the Development of Intelligent Diagnosis and Treatment in Traditional Chinese Medicine? A Survey and Empirical Analysis \[2025-02\] \[[paper](https://www.aminer.cn/pub/67bcdefd163c01c85045ed77/can-gpts-accelerate-the-development-of-intelligent-diagnosis-and-treatment-in-traditional)\]
    
6.  Large Language Models in Traditional Chinese Medicine: A Short Survey and Outlook \[2025-05\] \[[paper](https://www.aminer.cn/pub/68ac7b3b163c01c8508094d9/large-language-models-in-traditional-chinese-medicine-a-short-survey-and-outlook)\]
    

# 方法（开源研究）

注意：本仓库主要收集专注于中医药领域的大模型(TCM-LLM)，而非广泛的中文医疗大模型。以下“Chinese Medical（中文医疗大模型）”部分仅作参考，不属于本仓库的核心收集范围。

**Chinese Medical（中文医疗大模型）**

1.  **BenTsao** (original name: HuaTuo): Instruction-tuning Large Language Models With Chinese Medical Knowledge \[2023-04\] \[[paper](https://arxiv.org/abs/2304.06975)\] \[[repo](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)\]
2.  **Zhongjing**: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue \[2023-08\] \[AAAI 2024\] \[[paper](https://arxiv.org/abs/2308.03549)\] \[[repo](https://github.com/SupritYoung/Zhongjing)\]
3.  **HuatuoGPT**, Towards Taming Language Model to Be a Doctor \[2023-05\] \[EMNLP 2023\] \[[paper](https://arxiv.org/abs/2305.15075)\] \[[repo](https://github.com/FreedomIntelligence/HuatuoGPT)\]
4.  **HuatuoGPT-II**, One-stage Training for Medical Adaption of LLMs \[2023-11\] \[COLM 2024\] \[[paper](https://arxiv.org/abs/2311.09774)\] \[[repo](https://github.com/FreedomIntelligence/HuatuoGPT-II)\]
5.  ...

**TCM-LLM（中医药大模型）**

1.  **ShenNong-TCM**: A Traditional Chinese Medicine Large Language Model \[2023-06\] \[[repo](https://github.com/michael-wzhu/ShenNong-TCM-LLM)\]
    
2.  **CMLM-ZhongJing**: Large Language Model is Good Story Listener \[2023-06\] \[[paper](https://www.sciopen.com/article/10.26599/TST.2025.9010046)\] \[[repo](https://github.com/pariskang/CMLM-ZhongJing)\]
    
3.  **TCMLLM-PR**: evaluation of large language models for prescription recommendation in traditional Chinese medicine \[2023-08\] \[Digital Chinese Medicine\] \[[paper](https://www.sciencedirect.com/science/article/pii/S2589377725000072)\] \[[repo](https://github.com/2020MEAI/TCMLLM)\]
    
4.  **MedChatZH**: A tuning LLM for traditional Chinese medicine consultations \[2024-04\] \[Computers in Biology and Medicine\] \[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482524003743)\] \[[repo](https://github.com/tyang816/MedChatZH)\]
    
5.  **Huang-Di**大模型的构建 \[2024-01\] \[[paper](https://kns.cnki.net/kcms2/article/abstract?v=Ss1McYY34CcM4wcdUlqAPyzWC3gjGIJjs58usTGyX_wWc-yttCnA-kpGqPu-PTNtIT0HwIkKKiuLdLN4TToLAIJxl3ssHZsQujMs51CbksIg6TvzUDEo6fL-R8hurzzEgP5-XtDsrq_rgkqc7IETshQf5E4udJ8RiWTfpkbBjt5ztE-ku83r6A==&uniplatform=NZKPT&language=CHS)\] \[[repo](https://github.com/Zlasejd/HuangDI)\]
    
6.  **Qibo**: A Large Language Model for Traditional Chinese Medicine \[2024-03\] \[[paper](https://arxiv.org/abs/2403.16056)\]
    
7.  **PresRecST**: A novel herbal prescription recommendation algorithm for real-world patients with integration of syndrome differentiation and treatment planning \[2024-04\] \[JAMIA\] \[[paper](https://doi.org/10.1093/jamia/ocae066)\] \[[repo](https://github.com/2020MEAI/PresRecST)\]
    
8.  **MCM**: A Multi-Agent Collaborative Multimodal Framework For Traditional Chinese Medicine Diagnosis \[2024-06\] \[[paper](https://ieeexplore.ieee.org/document/11084334)\] \[[repo](https://github.com/JerryMazeyu/MCM)\]
    
9.  **Lingdan**: enhancing encoding of traditional Chinese medicine knowledge for clinical reasoning tasks with large language models \[2024-07\] \[JAMIA\] \[[paper](https://academic.oup.com/jamia/article-abstract/31/9/2019/7718082)\] \[[repo](https://github.com/TCMAI-BJTU/LingdanLLM)\]
    
10.  **BianCang**: A Traditional Chinese Medicine Large Language Model \[2024-11\] \[[paper](https://arxiv.org/abs/2411.11027)\] \[[repo](https://github.com/QLU-NLP/BianCang)\]
    
11.  **TCMChat**: A generative large language model for traditional Chinese medicine \[2024-12\] \[Pharmacological Research\] \[[paper](https://www.sciencedirect.com/science/article/pii/S1043661824004754)\] \[[repo](https://github.com/ZJUFanLab/TCMChat)\]
    
12.  **JingFang**: An Expert-Level Large Language Model for Traditional Chinese Medicine Clinical Consultation and Syndrome Differentiation-Based Treatment \[2025-02\] \[[paper](https://arxiv.org/abs/2502.04345)\]
    
13.  **OpenTCM**: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis \[2025-04\] \[BIGCOM25\] \[[paper](https://arxiv.org/abs/2504.20118)\] \[[repo](https://github.com/XY1123-TCM/OpenTCM)\]
    
14.  **TCM-KLLaMA**: Intelligent generation model for Traditional Chinese Medicine Prescriptions based on knowledge graph and large language model \[2025-05\] \[Computers in Biology and Medicine\] \[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482525002380)\]
    
15.  **Tianyi**: A Traditional Chinese Medicine all-rounder language model and its Real-World Clinical Practice \[2025-05\] \[[paper](https://arxiv.org/abs/2505.13156)\]
    
16.  **DoPI**: Doctor-like Proactive Interrogation LLM for Traditional Chinese Medicine \[2025-07\] \[[paper](https://arxiv.org/abs/2507.04877)\]
    
17.  **ShizhenGPT**: Towards Multimodal LLMs for Traditional Chinese Medicine \[2025-08\] \[[paper](https://arxiv.org/abs/2508.14706)\] \[[repo](https://github.com/FreedomIntelligence/ShizhenGPT)\]
    
18.  **ZhiFangDanTai**: Fine-tuning Graph-based Retrieval-Augmented Generation Model for Traditional Chinese Medicine Formula \[2025-09\] \[[paper](https://arxiv.org/abs/2509.05867)\]
    
19.  **TianHui**: A Domain-Specific Large Language Model for Diverse Traditional Chinese Medicine Scenarios \[2025-09\] \[[paper](https://arxiv.org/abs/2509.19834)\]


### ShenNong-TCM

简介：“神农”大模型，首个中医药大模型。

*   ChatMed\_TCM\_Dataset以我们开源的[中医药知识图谱](https://github.com/ywjawmw/TCM_KG)为基础；
    
*   采用以实体为中心的自指令方法，调用ChatGPT得到11w+的围绕中医药的指令数据；
    
*   ShenNong-TCM模型也是以LlaMA为底座，采用LoRA (rank=16)微调得到。
    

### CMLM-ZhongJing

简介：首个中医大语言模型——“仲景”。受古代中医学巨匠张仲景深邃智慧启迪，专为传统中医领域打造的预训练大语言模型。

### TCMLLM-PR

简介：面向处方推荐的大模型TCMLLM-PR。TCMLLM由北京交通大学计算机与信息技术学院医学智能团队开发的中医药大语言模型项目，项目针对中医临床智能诊疗问题中的处方推荐任务，通过整合真实世界临床病历、医学典籍与中医教科书等数据，构建了包含68k数据条目（共10M token）的处方推荐指令微调数据集，并使用此数据集，在ChatGLM大模型上进行大规模指令微调，最终得到了中医处方推荐大模型TCMLLM-PR。

### Huang-Di

简介：该名字源自中医古籍《黄帝内经》。首先在 Ziya-LLaMA-13B-V1基线模型的基础上加入中医教材、中医各类网站数据等语料库，训练出一个具有中医知识理解力的预训练语言模型（pre-trained model），之后在此基础上通过海量的中医古籍指令对话数据及通用指令数据进行有监督微调（SFT），使得模型具备中医古籍知识问答能力。

### MCM

简介：MCM(**M**ultimodal **C**hinese **M**edical LLM)是由上海计算机软件技术开发中心研发的多模态中医药问诊大模型，该模型可以通过与用户的对话进行问诊，问诊过程支持医学影像处理，问诊过程由**知识图谱驱动，具备可解释性**，同时支持常用中医知识问答。

简介2：为解决这些局限性，我们提出了 MCM，即一种用于中医诊断的多智能体协作多模态框架。该框架通过多智能体协作实现稳健且可解释的多模态诊断，为大型语言模型在中医领域的应用提供了新方法。

### LingdanLLM

简介：灵丹中医大模型，本项目旨在通过在百川 2 模型上继续预训练，构建一个更强大的基础模型，以支持下游任务的开发。我们特别关注中医药领域，因此选择了丰富的训练数据集，包括中医药古籍、教材和中国药典。这一过程不仅能增强模型对中医药知识的理解，还能为其深入掌握中医药理论和实践提供坚实的基础。通过本项目，我们希望显著提升人工智能在中医药领域的应用水平，为中医药的现代化做出贡献。

### BianCang

简介：扁仓中医大模型，扁仓是古代名医扁鹊、仓公的并称，泛指名医。我们期待扁仓中医大模型能够在延续中医传承和提升我国人民医疗健康水平方面做出一定的贡献。扁仓以Qwen2/2.5作为基座，采用先注入领域知识再进行知识激活和对齐的两阶段训练方法而得到。扁仓在中医辨病辨证等中医特色任务上取得了最先进的性能，并且在各种医学执照考试中表现优异。

### TCMChat

简介：在大规模精心整理的中医文本知识和中文问答（QA）数据集上进行了预训练（PT）和有监督微调（SFT）。具体而言，我们首先通过文本挖掘和人工验证，编制了一个包含六个中医场景的定制化数据集作为训练集，涉及中医知识库、选择题、阅读理解、实体提取、病例诊断以及草药或方剂推荐。接下来，我们以 Baichuan2–7B-Chat 作为基础模型，对该模型进行了预训练和有监督微调。

### JingFang

构建了 “经方”（JF）这一新型中医大型语言模型，该模型具备临床问诊和辨证的专业水平。我们提出了多智能体协作思维链机制（MACCTM），以实现全面且有针对性的临床问诊，使 “经方” 拥有高效准确的诊断能力。此外，还开发了辨证智能体和双阶段康复方案（DSRS），以精准提升辨证水平及后续的相应治疗效果。

### OpenTCM

OpenTCM 是一个基于知识图谱和大型语言模型（LLM，以 Kimi 为例）构建的中医智能问答 Web 应用。它旨在将结构化的中医知识与 LLM 的理解和生成能力相结合，为用户提供来源可靠、内容全面且易于理解的中医信息。该项目支持流式响应，以提升用户交互体验。

### TCM-KLLaMA

本文提出了一种借助中医知识图谱（KG）增强的高效方剂生成模型，称为 TCM-KLLaMA 模型。在该模型中，为 Chinese-LLaMA2-7B 模型配备了新的输出层和损失函数，以抑制幻觉并提高推荐准确性。研究构建了一个包含症状、舌诊和脉诊的中医知识图谱，并利用所提出的同义词与匹配知识注入（SMKI）机制对模型进行了微调。

### Tianyi

Tianyi采用7.6亿参数的语言模型，通过在包括经典文本、专家著作、临床记录和知识图谱等多种TCM语料库上进行预训练和微调，以系统性地学习和应用TCM知识。

### DoPI

DoPI系统采用了协作架构，包括一个指导模型和一个专家模型，指导模型通过多轮对话动态生成问题，专家模型利用中医深度知识给出诊断和治疗建议。

### ShizhenGPT

时珍 GPT 是首个专为中医药设计的多模态大语言模型。经过广泛训练，它在中医药知识方面表现出色，能够理解图像、声音、气味和脉象（支持望闻问切）。

### ZhiFangDanTai

通过融合图检索增强生成（GraphRAG）技术与大型语言模型微调，创建ZhiFangDanTai框架，利用图结构检索和整合中药知识，同时构建增强的指令数据集以提升模型整合检索信息的能力。

### TianHui

TianHui模型通过构建大规模的中医药语料库（包含0.97GB的无监督数据和611,312个问答对），并采用了两阶段训练策略，其中包括QLoRA、DeepSpeed Stage 2和Flash Attention 2技术。

# 方法（闭源产品）

### 岐黄问道·中医大模型

相关：[https://www.dajingtcm.com/product/3](https://www.dajingtcm.com/product/3)

时间：2023.08

机构：大经中医

简介：岐黄问道大模型是一个基于中医知识和数据的人工智能模型，由大经中医研发和发布。它可以根据用户提供的疾病、症状、体征等信息，给出中医的诊断和治疗方案，包括中药、食疗、茶饮、推拿、艾灸等多维度的养生调理建议。它的目的是实现中医临床诊疗和健康养生的智能化，传承和发展中医药文化。

### 聪宝素问

相关：[https://healthcare.tcmbrain.com/](https://healthcare.tcmbrain.com/)

时间：2023.08

机构：聪宝

简介：“聪宝素问 LLM5.1” 是深度优化的中医领域大模型，结合 DeepSeek 强大的蒸馏技术与私有数据微调，实现中医知识的深度挖掘与精准应用。

### 九为中药大模型

相关：[https://zhuanlan.zhihu.com/p/686578044](https://zhuanlan.zhihu.com/p/686578044)

时间：2024.01

机构：九为健康、华为云

简介：华为中医药大模型于2024年1月23日正式亮相。浙江九为健康科技股份有限公司与华为云计算技术有限公司在华为深圳总部签署了中医药大模型全面深化合作协议，共同推出了这一创新性的中医药大模型。

### 天河灵枢大模型

相关：[https://www.tj.gov.cn/sy/tjxw/202404/t20240411\_6596937.html](https://www.tj.gov.cn/sy/tjxw/202404/t20240411_6596937.html)

时间：2024.04

机构：国家超级计算天津中心、天津中医药大学

简介：天河灵枢大模型是由国家超级计算天津中心联合现代中医药海河实验室、天津中医药大学等机构，于2024年4月10日在第三届中医药高质量发展大会上发布的全球首个针灸领域专业大模型。该模型基于《灵枢》等中医经典著作，整合针灸临床循证证据库与知识图谱，通过对天河天元大模型进行专业化微调训练形成，具备个性化针灸方案生成、多维证据链溯源、古籍方剂智能耦合等核心功能。

### 本草智库·中药大模型

相关：[https://www.cdutcm.edu.cn/info/1074/15651.htm](https://www.cdutcm.edu.cn/info/1074/15651.htm)

时间：2024.05

机构：成都中医药大学

简介：本草智库大模型基于中国工程院院士、成都中医药大学首席教授陈士林团队本草基因组学的研究成果构建。本草智库汇集了1500万条中药材基原物种基因信息、3000余万条中药成分与靶点互作信息、400余万个化合物等中药研究底层核心数据，形成了覆盖中药全产业链的2000余万个实体和超20亿个关系对知识图谱，让中药材有了专属“基因身份证”。

### 数智本草

链接：[https://tcmaidd.tasly.com/ui/#/index](https://tcmaidd.tasly.com/ui/#/index)

时间：2024.05

机构：天士力医药集团

简介：数智本草大模型是由天士力医药集团与华为云共同开发的中医药领域专业大模型，于2024年5月9日在天津第四届中医药国际发展大会首次发布。该模型基于华为盘古大语言模型和盘古药物分子大模型，结合380亿参数量与中医药海量文本数据训练，拥有智能问答、交互计算和报告生成三种核心应用模式。通过整合现代科技与传统医学经验，其核心目标包括实现"从病到方"与"从方到病"的双向推导，以及推动中药组方优化、天然产物开发等领域的数智化转型。

### 海河·岐伯

相关：[https://www.hiascend.com/marketplace/solution/detail/2265](https://www.hiascend.com/marketplace/solution/detail/2265)

时间：2024.10

机构：天大智图

简介：“海河·岐伯”中医药大模型主要设计思路是建设以中医药人工智能大模型、中医药行业数据集为核心的数智化中医药产业大模型，打造人才培养、科研支持、辅助诊疗以及预防未病等应用场景。通过综合性的创新策略，该模型可推动中医药产业数智化转型，加速培育中医药产业新质生产力。

### 数智岐黄2.0

相关：[https://pharm.ecnu.edu.cn/08/27/c43775a657447/page.htm](https://pharm.ecnu.edu.cn/08/27/c43775a657447/page.htm)

时间：2024.11

机构：华东师范大学等

简介：数智岐黄2.0是由华东师范大学等多所机构联合开发的中医药领域多模态大模型，是一款基于AI技术与中医药、西医药深度融合的多模态大模型，拥有中医药知识智能问答、健康咨询、中医药知识图谱动态交互等功能。

### 中医横琴大模型

相关：[https://www.stdaily.com/web/gdxw/2025-06/20/content\_357526.html](https://www.stdaily.com/web/gdxw/2025-06/20/content_357526.html)

时间：2025.06

机构：横琴实验室

简介：中医横琴垂类大模型（Hengqin-R1）是由中医药广东省实验室（横琴实验室）中医药+AI智算中心团队研发的中医领域垂直大模型，通过整合中医经典文献、现代教材、学术成果及名医临床经验构建诊疗数据集与知识库，融合大模型精调、强化学习与知识图谱技术形成强推理能力。

# Citation

```
@misc{oyxy2025AwesomeTCMLLM,
    title = {Awesome-TCM-LLM: Traditional Chinese Medicine Large Language Model},
    author = {oyxy2019},
    year = {2025},
    url = {https://github.com/oyxy2019/Awesome-TCM-LLM},
}
```

