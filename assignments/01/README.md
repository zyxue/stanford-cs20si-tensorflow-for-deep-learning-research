# Instruction

https://docs.google.com/document/d/1CpBvivgmo_4sMmnlWqjFAXTXKa_bI9dJVSQLPZIu7Vg/edit


### Q2

**Notes on tf
[eager mode](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html)
and tf.placeholder**: they are not compatible (realized when using tf-1.6), and
I saw error like

```
   1741   """
   1742   if context.in_eager_mode():
-> 1743     raise RuntimeError("tf.placeholder() is not compatible with "
   1744                        "eager execution.")
   1745 

RuntimeError: tf.placeholder() is not compatible with eager execution.
```

The following results are quite reproducible.

Logistic regression on MNIST result:
```
$python q2_logistic_regression_MNIST_or_notMNIST.py mnist
/projects/btl2/zxue/miniconda3/envs/ml/lib/python3.5/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
data/mnist/train-images-idx3-ubyte.gz already exists
data/mnist/train-labels-idx1-ubyte.gz already exists
data/mnist/t10k-images-idx3-ubyte.gz already exists
data/mnist/t10k-labels-idx1-ubyte.gz already exists
Create training Dataset and batch it...
create testing Dataset and batch it...
Average loss epoch 0: 0.3653964425068955
Average loss epoch 1: 0.2937211207698944
Average loss epoch 2: 0.2842552832566028
Average loss epoch 3: 0.2796937625941842
Average loss epoch 4: 0.2740925426573254
Average loss epoch 5: 0.2753755952729735
Average loss epoch 6: 0.2694265393670215
Average loss epoch 7: 0.2682881838880306
Average loss epoch 8: 0.2676440643016682
Average loss epoch 9: 0.2693883784288584
Average loss epoch 10: 0.2623619491624278
Average loss epoch 11: 0.26178998572881834
Average loss epoch 12: 0.26181308315243834
Average loss epoch 13: 0.25951381734637324
Average loss epoch 14: 0.2614950296144153
Average loss epoch 15: 0.2608356742491556
Average loss epoch 16: 0.2580266118222891
Average loss epoch 17: 0.25712254742203755
Average loss epoch 18: 0.25907267188263494
Average loss epoch 19: 0.25833643565690795
Average loss epoch 20: 0.2561656144989091
Average loss epoch 21: 0.25658561661839485
Average loss epoch 22: 0.2550153321998064
Average loss epoch 23: 0.25499933062251223
Average loss epoch 24: 0.25737227066311724
Average loss epoch 25: 0.2522791906065026
Average loss epoch 26: 0.25218527209620145
Average loss epoch 27: 0.2542661675880122
Average loss epoch 28: 0.25225781930047414
Average loss epoch 29: 0.25422457706096563
Total time: 34.37617778778076 seconds
Accuracy 0.9234
```

Logistic regression on notMNIST result:

```
$python q2_logistic_regression_MNIST_or_notMNIST.py not_mnist
/projects/btl2/zxue/miniconda3/envs/ml/lib/python3.5/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Create training Dataset and batch it...
create testing Dataset and batch it...
Average loss epoch 0: 0.783124624365984
Average loss epoch 1: 0.7441129014935605
Average loss epoch 2: 0.7338698089122773
Average loss epoch 3: 0.7308982341095459
Average loss epoch 4: 0.7183866791946943
Average loss epoch 5: 0.7241070848564769
Average loss epoch 6: 0.7289994658425797
Average loss epoch 7: 0.7252381682395935
Average loss epoch 8: 0.711127616638361
Average loss epoch 9: 0.7196944094674532
Average loss epoch 10: 0.7183917911246765
Average loss epoch 11: 0.7237203316633092
Average loss epoch 12: 0.724167956793031
Average loss epoch 13: 0.7072137687095376
Average loss epoch 14: 0.7104390047317327
Average loss epoch 15: 0.7166374679914741
Average loss epoch 16: 0.7180519280738609
Average loss epoch 17: 0.7071109468160673
Average loss epoch 18: 0.7068922352652217
Average loss epoch 19: 0.7052256783080656
Average loss epoch 20: 0.7086321649163269
Average loss epoch 21: 0.7157147964072782
Average loss epoch 22: 0.6985762550387271
Average loss epoch 23: 0.7183989770883737
Average loss epoch 24: 0.7104743406523105
Average loss epoch 25: 0.7188465511383012
Average loss epoch 26: 0.7044777250567148
Average loss epoch 27: 0.7098890296248502
Average loss epoch 28: 0.7086790768906127
Average loss epoch 29: 0.7091376062742499
Total time: 34.40691137313843 seconds
Accuracy 0.872
```

To visualize the graph:

```
tensorboard --port 22222 --logdir ./graphs/
````

A screenshot:

<img src="https://github.com/zyxue/stanford-cs20si-tensorflow-for-deep-learning-research/blob/master/assignments/01/q2_graph_logreg.png?raw=true" />
