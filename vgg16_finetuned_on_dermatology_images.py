from __future__ import print_function, division
import torch
from torchvision import transforms, models
import pandas as pd
import numpy as np
import os
import skimage
from skimage import io
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim import lr_scheduler
import time
import copy
import sys
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def train_model(label, dataloaders, device, dataset_sizes, model,
                criterion, optimizer, scheduler, num_epochs=2, is_inception=False):
    since = time.time()
    training_results = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            print(phase)
            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch["image"].to(device)
                labels = batch[label]
                labels = torch.from_numpy(np.asarray(labels)).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.float()
                    if is_inception:
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            # print("Loss: {}/{}".format(running_loss, dataset_sizes[phase]))
            print("Accuracy: {}/{}".format(running_corrects,
                                           dataset_sizes[phase]))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            training_results.append([phase, epoch, epoch_loss, epoch_acc])
            # deep copy the model
            if epoch > 10:
                if phase == 'val' and epoch_loss < best_loss:
                    print("New leading accuracy: {}".format(epoch_acc))
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'val':
                best_loss = epoch_loss
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    training_results = pd.DataFrame(training_results)
    training_results.columns = ["phase", "epoch", "loss", "accuracy"]
    return model, training_results


class SkinDataset():
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'filename'])
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        multi_labels = self.df.loc[self.df.index[idx], 'numeric_label']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
        label_0 = self.df.loc[self.df.index[idx], 'label_0']
        label_1 = self.df.loc[self.df.index[idx], 'label_1']
        label_2 = self.df.loc[self.df.index[idx], 'label_2']
        label_3 = self.df.loc[self.df.index[idx], 'label_3']
        label_4 = self.df.loc[self.df.index[idx], 'label_4']
        label_5 = self.df.loc[self.df.index[idx], 'label_5']
        label_6 = self.df.loc[self.df.index[idx], 'label_6']
        label_7 = self.df.loc[self.df.index[idx], 'label_7']
        label_8 = self.df.loc[self.df.index[idx], 'label_8']
        if self.transform:
            image = self.transform(image)
        sample = {
                    'image': image,
                    'label_0': label_0,
                    'label_1': label_1,
                    'label_2': label_2,
                    'label_3': label_3,
                    'label_4': label_4,
                    'label_5': label_5,
                    'label_6': label_6,
                    'label_7': label_7,
                    'label_8': label_8,
                    'multi_label': multi_labels,
                    'fitzpatrick': fitzpatrick,
                    'hasher': hasher
                }
        return sample


def custom_load(
        batch_size=256,
        num_workers=20,
        train_dir='',
        val_dir='',
        image_dir='/mas/projects/dermatology-data/grohdata'):
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    class_sample_count = np.array(train[label].value_counts().sort_index())
    print(class_sample_count)
    class_sample_count[8] = 1.5*class_sample_count[8]
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train[label]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type('torch.DoubleTensor'),
        len(samples_weight),
        replacement=True)
    dataset_sizes = {"train": train.shape[0], "val": val.shape[0]}
    transformed_train = SkinDataset(
        csv_file=train_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # change to 299+32 for inception
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards # change to 299 for inception
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])
        )
    transformed_test = SkinDataset(
        csv_file=val_dir,
        root_dir=image_dir,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        )
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # shuffle=True,
            num_workers=num_workers),
        "val": torch.utils.data.DataLoader(
            transformed_test,
            batch_size=batch_size,
            # sampler=sampler,
            shuffle=False,
            num_workers=num_workers)
        }
    return dataloaders, dataset_sizes


if __name__ == '__main__':
    print("\nPlease specify number of epochs and 'dev' mode or not... e.g. python train.py 10 full \n")
    n_epochs = int(sys.argv[1])
    dev_mode = sys.argv[2]
    holdout = sys.argv[3]
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_mode == "dev":
        df = pd.read_csv("/mas/projects/dermatology-data/a_labeled_data.csv").sample(1000)
    else:
        df = pd.read_csv("/mas/projects/dermatology-data/a_labeled_data.csv")
        df2 = pd.read_csv("/mas/projects/dermatology-data/a_website_selection_confirmed_by_dermatologist_1.csv")
        df2 = df2[["hasher","expert_label","expert_fitzpatrick","new_addition"]]
        df2.columns = ["hasher", "condition_expert", "fitzpatrick_expert", "new_addition"]
        df2 = df2[["condition_expert", "fitzpatrick_expert", "hasher"]]
        df2["expert_confirmed"] = 1
        df2.hasher = df2.hasher.str.replace(".jpg","")
        df2 = df2[df2.hasher.duplicated()==False]
        df3 = pd.merge(df,df2,on="hasher",how="outer")
        df3[df3['fitzpatrick']!=df3.fitzpatrick_expert].fitzpatrick = df3[df3['fitzpatrick']!=df3.fitzpatrick_expert].fitzpatrick_expert
        df3.expert_confirmed = df3.expert_confirmed.fillna(0)
        df3 = df3[df3.hasher.duplicated()==False]
        df3.loc[df3.hasher=="eb1af58b67f0f61d19c872453c26ee21", "hasher"] = 'eff7a873ad735f3976b551d69d6c7e76'
        df3.loc[df3.filename.isnull(), "filename"] = df3.hasher + ".jpg"
        df3.loc[df3.condition_expert=="Lyme Disease", "numeric_label"] = 6
        df3.loc[df3.condition_expert=="Syphilis", "numeric_label"] = 7
        b = pd.read_csv("/mas/projects/dermatology-data/b_fitzpatrick17k.csv")
        b = b[["hasher","fitzpatrick","category2", "qc"]]
        b["additional"] = 1
        b.hasher = b.hasher.str.replace(".jpg","")
        df3 = df3.append(b)
        df3 = df3[df3.hasher.duplicated()==False]
        df3.loc[(df3.additional==1) & (df3.qc=="1 Diagnostic") & (df3.category2=="inflammatory") & ((df3.fitzpatrick==5) | (df3.fitzpatrick==6) | (df3.fitzpatrick==1) | (df3.fitzpatrick==2)), "expert_confirmed"] = 1
        df = df3
        print(df.shape)

    print(df.columns)
    print(df['fitzpatrick'].value_counts())
    df.numeric_label = df.numeric_label.fillna("8")
    df["numeric_label"] = df["numeric_label"].astype(int)
    df["multi_label"] = df["numeric_label"]
    print(df.multi_label.value_counts())
    df.multi_label = df.multi_label.fillna(8)
    df = df.fillna(0)
    df['atopic_dermatitis'] = df.atopic_dermatitis.astype(int)
    df['lichen_planus'] = df.lichen_planus.astype(int)
    df["dermatomyositis"] = df["dermatomyositis"].astype(int)
    df["pityriasis rosea"] = df["pityriasis rosea"].astype(int)
    df["pityriasis_rubra_pilaris"] = df["pityriasis_rubra_pilaris"].astype(int)
    df["cutaneous_t-cell_lymphoma"] = df["cutaneous_t-cell_lymphoma"].astype(int)
    df["lyme"] = df["lyme"].astype(int)
    df["secondary_syphilis"] = df["secondary_syphilis"].astype(int)
    df["label_0"] = df['atopic_dermatitis']
    df["label_1"] = df['lichen_planus']
    df["label_2"] = df["dermatomyositis"]
    df["label_3"] = df["pityriasis rosea"]
    df["label_4"] = df["pityriasis_rubra_pilaris"]
    df["label_5"] = df["cutaneous_t-cell_lymphoma"]
    df["label_6"] = df["lyme"]
    df["label_7"] = df["secondary_syphilis"]
    df["label_8"] = df["additional"]
    df["filename"] = df["hasher"] + ".jpg"
    print("Rows: {}".format(df.shape[0]))
    print("Atopic Dermatitis 1: {}".format(df[df.numeric_label == 1].shape[0]))
    print("Lichen Planus 2: {}".format(df[df.numeric_label == 2].shape[0]))
    print("Dermatomyositis 3: {}".format(df[df.numeric_label == 3].shape[0]))
    print("Pityriasis Rosea 4: {}".format(df[df.numeric_label == 4].shape[0]))
    print("Pityriasis Rubra Pilaris 5: {}".format(
        df[df.numeric_label == 5].shape[0]))
    print("Cutaneous T-Cell Lymphoma 6: {}".format(
        df[df.numeric_label == 6].shape[0]))
    print("Lyme 7: {}".format(df[df.numeric_label == 7].shape[0]))
    print("Secondary Syphilis 8: {} \n".format(df[df.numeric_label == 8].shape[0]))
    if holdout == "random":
        train, test, y_train, y_test = train_test_split(
                                            df,
                                            df.numeric_label,
                                            test_size=0.2,
                                            random_state=429,
                                            stratify=df.numeric_label)
    elif holdout == "expert":
        train = df[df.expert_confirmed==0]
        test = df[df.expert_confirmed==1]
    print(test.shape)
    print(test.shape)
    train_path = "/mas/projects/dermatology-data/trainer/train.csv"
    test_path = "/mas/projects/dermatology-data/trainer/test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print("Training Shape: {}, Test Shape: {} \n".format(
    train.shape,
    test.shape)
    )
    print(train.columns)
    train["multi_label"] = train["numeric_label"]
    for indexer, label in enumerate([
                                    "multi_label"
                                    ]):
        print(label)
        print(train[label].value_counts().sort_index())
        label_codes = sorted(list(train[label].unique()))
        print(label_codes)
        dataloaders, dataset_sizes = custom_load(
            256,
            20,
            "{}".format(train_path),
            "{}".format(test_path))
        model_ft = models.vgg11(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, len(label_codes)),                   
                      nn.LogSoftmax(dim=1))
        is_inception=False
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model_ft.parameters())
        print('{} total parameters'.format(total_params))
        total_trainable_params = sum(
            p.numel() for p in model_ft.parameters() if p.requires_grad)
        print('{} total trainable parameters'.format(total_trainable_params))
        model_ft = model_ft.to(device)
        model_ft = nn.DataParallel(model_ft)
        criterion = nn.NLLLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
        print("\nTraining classifier for {}........ \n".format(label))
        print("....... processing ........ \n")
        model_ft, training_results = train_model(
            label,
            dataloaders, device,
            dataset_sizes, model_ft,
            criterion, optimizer_ft,
            "exp_lr_scheduler", n_epochs, is_inception)
        torch.save(model_ft.state_dict(), "/mas/projects/dermatology-data/inflammatory.pth")

        print("Training Complete")
        training_results.to_csv("/mas/projects/dermatology-data/results/training_{}_{}_{}.csv".format(n_epochs, label, holdout))
        model = model_ft.eval()
        loader = dataloaders["val"]
        prediction_list = []
        fitzpatrick_list = []
        hasher_list = []
        labels_list = []
        multi_labels_list = []
        p_list = []

        topk_p = []
        topk_n = []
        d1 = []
        d2 = []
        d3 = []
        p1 = []
        p2 = []
        p3 = []
        with torch.no_grad():
            running_corrects = 0
            for i, batch in enumerate(dataloaders['val']):
                inputs = batch["image"].to(device)
                classes = batch[label].to(device)
                multi_classes = batch["multi_label"].to(device)
                fitzpatrick = batch["fitzpatrick"]
                hasher = batch["hasher"]
                outputs = model(inputs.float())
                probability = outputs
                ppp, preds = torch.topk(probability, 1)
                if label == "multi_label":
                    _, preds5 = torch.topk(probability, 3)
                    topk_p.append(_.cpu().tolist())
                    topk_n.append(preds5.cpu().tolist())
                # for t, p in zip(classes.view(-1), preds.view(-1)):
                #     confusion_matrix[t.long(), p.long()] += 1
                running_corrects += torch.sum(preds == classes.data)
                p_list.append(ppp.cpu().tolist())
                prediction_list.append(preds.cpu().tolist())
                labels_list.append(classes.tolist())
                multi_labels_list.append(multi_classes.tolist())
                fitzpatrick_list.append(fitzpatrick.tolist())
                hasher_list.append(hasher)
            acc = float(running_corrects)/float(dataset_sizes['val'])
        if label == "multi_label":
            for j in topk_n:
                for i in j:
                    print(i)
                    d1.append(i[0])
                    d2.append(i[1])
                    d3.append(i[2])
            for j in topk_p:
                for i in j:
                    p1.append(i[0])
                    p2.append(i[1])
                    p3.append(i[2])
            df_x=pd.DataFrame({
                                "hasher": flatten(hasher_list),
                                "label": flatten(labels_list),
                                "multiclass_label": flatten(multi_labels_list),
                                "fitzpatrick": flatten(fitzpatrick_list),
                                "prediction_probability": flatten(p_list),
                                "prediction": flatten(prediction_list),
                                "d1": d1,
                                "d2": d2,
                                "d3": d3,
                                "p1": p1,
                                "p2": p2,
                                "p3": p3})
        else:
            df_x=pd.DataFrame({
                                "hasher": flatten(hasher_list),
                                "label": flatten(labels_list),
                                "multiclass_label": flatten(multi_labels_list),
                                "fitzpatrick": flatten(fitzpatrick_list),
                                "prediction_probability": flatten(p_list),
                                "prediction": flatten(prediction_list)})
        df_x.to_csv("/mas/projects/dermatology-data/results/results_{}_{}_{}.csv".format(n_epochs, label, holdout),
                        index=False)
        print("\n Accuracy: {} \n".format(acc))
    print("done")
