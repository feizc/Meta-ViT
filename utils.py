import torch 
import math 
import time 
import pickle 
import os 


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits



def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    #model = model.cuda()
    #devices = [x for x in range(torch.cuda.device_count())]
    #return torch.nn.DataParallel(model,  device_ids=devices) 
    return model 


def test_model_on_dataset(model, test_loader, device, save_logits=False, model_id=0):

    model.eval() 
    logits_list = []
    
    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        end = time.time()
            # assert to make sure the imagenet held-out minival logic is consistent across machines.
            # tested on a few machines but if this fails for you please submit an issue and we will resolve.
            # assert dataset.train_dataset.__getitem__(dataset.sampler.indices[1000])['image_paths'].endswith('n01675722_4108.JPEG')

        for i, batch in enumerate(test_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            data_time = time.time() - end
            y = labels
            
            logits = model(inputs)

            if save_logits == True: 
                item = {
                    'logits': logits.detach().cpu(),
                    'label': labels.detach().cpu()
                }
                logits_list.append(item) 

            pred = logits.argmax(dim=1, keepdim=True).to(device)
            
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0) 

            batch_time = time.time() - end
            end = time.time()
            if (i+1) % 20 == 0:
                percent_complete = 100.0 * i / len(test_loader)
                print(
                    f"[{percent_complete:.0f}% {i}/{len(test_loader)}]\t"
                    f"Acc: {100 * (correct/n):.2f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                )
                break 

        if save_logits == True: 
            target_file = os.path.join('./ensemble', 'model_'+str(model_id)+'.pickle') 
            with open(target_file, 'wb') as f: 
                pickle.dump(logits_list, f)

        top1 = correct / n
        return top1



