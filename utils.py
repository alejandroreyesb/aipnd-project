
import torch
from torchvision import datasets, transforms, models
from PIL import Image


import os

def my_dataloader(data_dir):
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'training':transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation':transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testing':transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training':datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'validation':datasets.ImageFolder(valid_dir, transform=data_transforms['validation']),
        'testing':datasets.ImageFolder(test_dir, transform=data_transforms['testing'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # Dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(
            image_datasets['training'],
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        ),
        'validation': torch.utils.data.DataLoader(
            image_datasets['validation'],
            batch_size=64,
            num_workers=2,
            pin_memory=True
        ),
        'testing': torch.utils.data.DataLoader(
            image_datasets['testing'],
            batch_size=64,
            num_workers=2,
            pin_memory=True
        )
    }
    
    return image_datasets, dataloaders
    

def my_training(model, criterion, epochs, dataloaders, optimizer, gpu):
    
    # Use GPU if it's available
    device_found = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (device_found.type == 'cuda'):
        if (gpu):
            print ('using gpu as requested')
        else:
            print ('Your device support GPU but I will use cpu as requested')
            device_found = "cpu"
    else:
        if (gpu):
            print('Your device cant use gpu, using cpu instead')            
        else:
            print('using cpu as requested')
    
    device = device_found
    
    steps = 0
    print_every = 50
    model.to(device)
    
    print(f"Training on {device} started")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['training']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            #logps = model.forward(inputs)
            logps = model(inputs)
            loss = criterion(logps, labels)       
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['validation']:
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                         
                # Calculate accuracy  
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1) 
                equals = top_class == labels.view(*top_class.shape)
                accuracy += equals.float().mean().item()
        
        print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {valid_loss/len(dataloaders['validation']):.3f}.. "
                          f"Test accuracy: {accuracy/len(dataloaders['validation']):.3f}")

    print("Training complete!")
    
    return model


def my_save_checkpoint(model, image_datasets, epochs, optimizer, save_dir, architecture):
    """Save the model checkpoint."""
    model.class_to_idx = image_datasets['training'].class_to_idx

    checkpoint = {
        'arch': architecture,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_transform = transforms.Compose([transforms.Resize(256), 
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    # open image and apply 
    pil_image = Image.open(image)
    pil_image = pil_transform(pil_image)

    return pil_image
    
    
def my_predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    
    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()

    # Move model to the same device as the image
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        
    model = model.to(device)
    image = image.to(device)

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        logps = model.forward(image)

    probability = torch.exp(logps)

    # Get top K probabilities and indices
    top_prob, top_indices = probability.topk(topk, dim=1)

    # Invert the class_to_idx dictionary
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    top_indices_list = top_indices.cpu().tolist()[0] 
    
    #top_classes = [idx_to_class[idx] for idx in top_indices]
    top_classes = []
    for index in top_indices_list:
        top_classes.append(idx_to_class[index])

    return top_prob.cpu().numpy()[0], top_classes


def my_load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    print('Architecture loaded')
    
    arch = checkpoint['arch']
    #print(arch)
    #print(checkpoint.get('arch', 'vgg16'))
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        raise ValueError(f'Arquitectura no soportada: {arch}')
    
    #model = models.vgg16(pretrained=True)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Load the classifier
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model