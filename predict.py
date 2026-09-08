#python predict.py /path/to/image checkpoint
#Opciones: 
#* Devolver las K clases principales: python predict.py input checkpoint --top_k 3 
#* Usar un mapeo de nombres de categorías reales: python predict.py input checkpoint --category_names cat_to_name.json 
#* Usar GPU para la inferencia: python predict.py input checkpoint --gpu


import argparse
import torch

from utils import my_predict, my_load_checkpoint
from torch import optim, nn
#from torchvision import models

import json

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('image_path',      type=str, default='./flowers/test/10/image_07104.jpg', help='Specify image location')
    parser.add_argument('checkpoint',      type=str, default='./checkpoint.pth', help='Specify the checkpoint')
    parser.add_argument('--topk',          type=int, default='1',   help='Specify the top k most classes to return')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Specify the use of gpu power over cpu')
    
    # paarsing arguments
    args = parser.parse_args()

    print(args)    
    
    checkpoint_path = args.checkpoint
    topk = args.topk
    #category_names = args.category_names
    #gpu = args.gpu
    
    #Aqui 
    model = my_load_checkpoint(checkpoint_path)
    
    #print(model)
    
    #image_path, model, topk
    probs, classes = my_predict(args.image_path, model, topk, args.gpu)
    print(f"Probabilities: {probs}")
    print(f"Classes: {classes}")
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            for label, prob in zip(classes, probs):
                print('{:.4f} / {} '.format(prob, cat_to_name[label]))
    else:
        for label, prob in zip(classes, probs):
            print('{:.4f} / {} '.format(prob, label))

    

if __name__ == '__main__':
    main()





