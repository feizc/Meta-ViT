import argparse
import pickle
import os


import clip 
import torch 
import torch.nn as nn 


from datasets import cifa100_data_load 
from utils import test_model_on_dataset, get_model_from_sd 


device = "cuda" if torch.cuda.is_available() else "cpu"  


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('./data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('./ckpt/vit'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--ensemble",
        type=bool,
        default=True,
    )
    return parser.parse_args()



if __name__ == '__main__': 
    args = parse_arguments() 
    NUM_MODELS = 72 
    
    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)] 
    base_model, preprocess = clip.load('./ckpt/clip/ViT-B-32.pt', 'cpu', jit=False) 

    for j, model_path in enumerate(model_paths):
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu')) 
        model = get_model_from_sd(state_dict, base_model) 
        model = model.to(device) 

        _, val_loader = cifa100_data_load() 
        accuracy = test_model_on_dataset(model, val_loader, device, save_logits=args.ensemble, model_id=j) 
        print(accuracy) 
    

    nnf_softmax = nn.Softmax(dim=1) 

    if args.ensemble == True: 
        file_list = os.listdir('./ensemble') 
        result_logits_list = []
        for file in file_list: 
            file_path = os.path.join('./ensemble', file) 
            with open(file_path, 'rb') as f: 
                data = pickle.load(f)
            result_logits_list.append(data) 
        
        correct = 0 
        total_num = 0 
        # ensemble for total dataset 
        for i in range(len(result_logits_list[0])): 
            logits = nnf_softmax(result_logits_list[0][i]['logits'] )
            labels = result_logits_list[0][i]['label']
            for j in range(1, NUM_MODELS): 
                logits += nnf_softmax(result_logits_list[j][i]['logits'] )
                assert labels.equal(result_logits_list[j][i]['label'] ) 
            pred = logits.argmax(dim=1, keepdim=True) 
            correct += pred.eq(labels.view_as(pred)).sum().item() 
            total_num += labels.size(0) 
        print(correct / float(total_num))


