# genshin-enlightenment-selection

An exact strategy to select the optimal artifact to use the Dust of Enlightenment.



## Requirements:

numpy; matplotlib, pandas



## 用法

`python artifacts_selection.py`



## 参数（config.json文件内容）

`json_dir`： 使用[Yas项目](https://github.com/wormtql/yas)得到的圣遗物json文件地址



`sand`：圣遗物沙漏主词条选项



`cup`：圣遗物杯子主词条选项



`head`：圣遗物头冠主词条选项



（别问为什么翻译这么抽象，因为Yas得到的json文件就是这么抽象）



`ass_reward_dic`：各个圣遗物副词条单个词条对应分数



`init_vals`：假想中的圣遗物初始词条数（这个不能从mona json文件中得到，所以不建议更改）



`artifact_name`：最想要提升的圣遗物套装名（不知道的话就去翻json文件）



`smooth_bandwidth`：绘图超参数，可以适当光滑P.M.F.



`short_target`：短期策略的目标
