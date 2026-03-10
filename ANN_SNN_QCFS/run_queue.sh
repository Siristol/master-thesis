
python main_train.py -b=160 --dataset=cifar10 --model=mobilenet -lr=0.1 --epochs=200 -L=8 -wd=0.0005

python main_train.py -b=160 --dataset=cifar100 --model=mobilenet -lr=0.05 --epochs=200 -L=8 -wd=0.0005

python main_train.py -b=32 --dataset=coco --model=mobilenet -lr=0.001 --epochs=50 -L=8 -wd=0.0005
