# Pytorch量化学习与FlexGen量化结合
* pytorch >=1.7
* torchvision >=0.5

## 量化存储？

## 量化计算？

## Pytorch对量化的支持
### 1. Quantized Tensor：
scale和zero_point两个参数建立起了fp tensor(x)到量化tensor(xq)的映射关系：
```
xq = round(x / scale + zero_point)
```
在PyTorch中，选择合适的scale和zero_point的工作就由各种observer来完成.

### 2. 三种方式：
* Post Training Dynamic Quantization
    - torch.quantization.quantize_dynamic,只有权重被量化。
    - observer用来根据四元组（min_val，max_val，qmin, qmax）来计算2个量化的参数：scale和zero_point。qmin、qmax是算法提前确定好的，min_val和max_val是从输入数据中观察到的，所以起名叫observer
    - 提前把模型中某些op的参数(weight)量化为INT8（静态的），然后在运行的时候**动态**的把输入(input)量化为INT8，然后在当前op输出的时候再把结果requantization回到float32类型。动态量化默认只适用于Linear以及RNN的变种。
* Post Training Static Quantization
    - 对比Dynamic Quantization，Static Quantization包含activation即post process。因为静态量化的前向推理过程始终都是使用INT计算的，静态量化过程：
        + fuse_model:合并一些可以合并的layer，提高速度和准确度
        + 设置qconfig：设置量化observer
        + prepare：用于收集和定标数据,module的_forward_hooks上都被插入了activation的HistogramObserver，当这些子module计算完毕后，结果会被立刻送到其_forward_hooks中的HistogramObserver进行观察。
        + 喂数据，获取数据的分布特点来计算activation的scale和zero_point。
        + 转换模型，和Dynamic类似，使用字典替换，在使用from_float() API将模型参数转换为INT.
* QAT（Quantization Aware Training）
    - 在训练中就开启量化，同样也是5步走。
        + 设置qconfig:设置qconfig前将模型设置为训练模式。
        + fuse_modules
        + prepare_qat：比静态量化多做两步。
        + 喂数据：和静态量化完全不同。在训练过程中，fake_quants发挥作用，来替换静态量化的效果。
        + 转换


* pytorch量化计算的原理。
* flexgen量化存储的原理。