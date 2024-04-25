import argparse                                
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os
import random
from devil import load_trained_model, load_categories_mapping

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint', action='store', default='model_checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='3')
    parser.add_argument('--image_path', dest='image_path', default='flowers/test/1/image_06743.jpg')
    parser.add_argument('--category_mapping', dest='category_mapping', default='category_mapping.json')
    parser.add_argument('--device', action='store', default='cpu')
    return parser.parse_args()

def process_image(image):
    ''' Process a PIL image for use in a PyTorch model: resize, crop, and normalize. '''
    
    img_pil = Image.open(image)
   
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = preprocess(img_pil)
    
    return image

def make_prediction(image_path, model, topk=3, device='cpu'):
    ''' Predict the class probabilities of an image using a trained model. '''
    
    if device == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
        
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if device == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    probs = np.array(probability.topk(topk)[0][0])
    
    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes

def main(): 
    args = get_input_args()
    device = args.device
    model = load_trained_model(args.model_checkpoint)
    category_mapping = load_categories_mapping(args.category_mapping)
    
    image_path = args.image_path
    probs, classes = make_prediction(image_path, model, int(args.top_k), device)
    labels = [category_mapping[str(index)] for index in classes]
    probabilities = probs
    
    print('Selected File: ' + image_path)
    print(labels)
    print(probabilities)
    
    i = 0 
    while i < len(labels):
        print("{} with a probability of {}".format(labels[i], probabilities[i]))
        i += 1 

if __name__ == "__main__":
    main()
