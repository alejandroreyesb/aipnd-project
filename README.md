# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

You can se the html to see how it looks in the jupyter notebook. 

To create the enviroment in your computer with CUDA I use this command: 

pip install torch torchvision torchaudio transformers --index-url https://download.pytorch.org/whl/nightly/cu130

of course you need:

conda install jupyter notebook

##To run in command line you have the options:

##To train:
python train.py data_dir --save_dir --arch --learning_rate --hidden_units --epochs --gpu

| Parameter | Description |
| --------- | ------------|
|data_dir   |       'Specify the image data directory'|  
|--save_dir  |      'Specify directory to save checkpoints'|  
|--arch      |      'Specify the model architecture default=\'vgg16\' also supported vgg13 and vgg19'|  
|--learning_rate|   'Specify the laerning rate for your model default=0.01'|  
|--hidden_units |  'Specify the hidden units of your model default=512'|  
|--epochs |        'Specify the number of epochs default=20'|  
|--gpu    |         'Specify the use of gpu power over cpu'|  

example:  python train.py flowers --epochs 10 --gpu  


##To test:

python predict.py  image_path checkpoint --topk --category_names --gpu

