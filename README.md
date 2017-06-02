KDD Travel time 排名28/3574的解决方案
================================

共享一下自己的代码， 主要用了grid search 5 folds CV + 1 fold eval set，因为到了>后期cv grid search出来的参数也出现overfit了。所以第6 fold的cv没有参与最优的搜索，这是作为eval作用。最终排名28/3574，一个人的写的程序，作为参考。

入口程序

```
cd kdd2017
./bin/compute_kdd2017 config/config.json
```

KDD travel time competition rank 28/3574 solution
======================================


Share my code here. Basically I have used 5-fold cv grid search. Finally I got over-fitting , so I added one more fold for evaluation (not for grid search). At the end, I got a rank of 28/3574. Single person work. Hope it would be useful for you. 

how to start

```
cd kdd2017
./bin/compute_kdd2017 config/config.json
```
