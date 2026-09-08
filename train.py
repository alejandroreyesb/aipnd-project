#Uso básico: python train.py data_directory
#Muestra la pérdida de entrenamiento, pérdida de validación y precisión de validación mientras la red se entrena
#Opciones: 
# * Establecer directorio para guardar checkpoints: python train.py data_dir --save_dir save_directory 
# * Elegir arquitectura: python train.py data_dir --arch "vgg16" 
# * Configurar hiperparámetros: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
# * Usar GPU para el entrenamiento: python train.py data_dir --gpu
#
#

import argparse
import torch

from utils import my_dataloader, my_training, my_save_checkpoint, my_predict
from torch import optim, nn
from torchvision import models


def main():
    # argument object
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir',        type=str, default='./flowers/', help='Specify the image data directory')
    parser.add_argument('--save_dir',      type=str, default='./', help='Specify directory to save checkpoints')
    parser.add_argument('--arch',          type=str, default='vgg16',   help='Specify the model architecture default=\'vgg13\'')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Specify the laerning rate for your model')
    parser.add_argument('--hidden_units',  type=int, default=512, help='Specify the hidden units of your model')
    parser.add_argument('--epochs',        type=int, default=20, help='Specify the number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Specify the use of gpu power over cpu')

    # paarsing arguments
    args = parser.parse_args()

    print(args)

    #data_dir = args.data_dir
    save_dir = args.save_dir
    learning_rate = args.learning_rate
    architecture = args.arch
    hidden_units = args.hidden_units
    epochs = args.epochs
    #gpu = args.gpu

    # Load data
    print(f"Loading data from {args.data_dir}")
    image_datasets, dataloaders = my_dataloader(args.data_dir)

    print(f"Training samples: {len(image_datasets['training'])}")
    print(f"Validation samples: {len(image_datasets['validation'])}")
    print(f"Test samples: {len(image_datasets['testing'])}")
    
    

    # A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen
    if architecture.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture.lower() == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("model not valid try to use vgg13, vgg16 or vgg19")
        return 
    
     # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    initial_features = model.classifier[0].in_features
    
    #hidden_units
    #Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    model.classifier = nn.Sequential(nn.Linear(initial_features, 3072),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(3072, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    
    #learning_rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    

    #Paramas(epochs, dataloaders, learning_rate, hidden_units, device):
    model = my_training(model, criterion, epochs, dataloaders, optimizer, args.gpu)

    #print(model)
    #model, image_datasets, epochs, optimizer, save_dir
    my_save_checkpoint(model, image_datasets, epochs, optimizer, save_dir, architecture)
    
    

if __name__ == '__main__':
    main()