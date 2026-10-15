"""OGC 数字人情感陪伴对话模型 —— 数据、训练、推理、评分工具链。

模块职责（改动请遵守这条边界）：

- :mod:`label_space`      官方标签枚举的唯一真相来源，导出 ``configs/label_space.json``
- :mod:`data_processor`   数据适配器：对话级 -> 逐预测点样本，统一 history/target 契约
- :mod:`evaluate`         本地离线评分：情绪 / 画像 / BLEU-4 / ROUGE-L / BERTScore
- :mod:`validate_submission`  提交前的 schema 校验器
- :mod:`cli`              上述能力的统一命令行入口

约定：本包内不得出现网络调用。评测容器完全离线，任何 ``requests`` /
``huggingface_hub.download`` 都会导致正式测评分归零。
"""

__all__ = ["label_space", "data_processor", "evaluate", "validate_submission"]
