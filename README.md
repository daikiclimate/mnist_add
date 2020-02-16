# mnist_add

環境

Pytorch 1.1など

最近WANDBを使い始めたので、一部それも含まれている

Utils

・savefig.ipynb
今回はpytorchのmnist画像をわざわざ保存している。
基本的には触る必要はないと思います。

・make_pair.py
保存した画像からペアのリストを作成する。
data/annotation/train_pair.txt
data/annotation/test_pair.txt
にペアを保存している

main Code
・Net.py
ネットワーク構造

・Mydataset.py
pytorchのデータセットクラスを作成

・train.py
~~初期検討の段階なので、あまりまとまってない~~　汚いコードですいません。
ハイパラ探索もする予定はないので、適当目です。
WANDB = 1にすると、WANDBに結果を転送してグラフ化することができます。

