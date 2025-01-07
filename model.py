import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm
import kagglehub

# Veri setini indirme
path = kagglehub.dataset_download(paramaggarwalfashion-product-images-dataset)
print(Path to dataset files, path)

# Dosya yolları
BASE_DIR = root.cachekagglehubdatasetsparamaggarwalfashion-product-images-datasetversions1fashion-dataset
CSV_FILE = os.path.join(BASE_DIR, styles.csv)
IMAGES_DIR = os.path.join(BASE_DIR, images)

# Veri setini yükleme ve gerekli sütunları filtreleme
df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
df = df[[id, season, year, usage, articleType]].dropna()
df[image_path] = df[id].astype(str) + .jpg
df = df[df[image_path].apply(lambda x os.path.exists(os.path.join(IMAGES_DIR, x)))]

# Sınıfları encode etme
label_encoders = {}
for col in [season, year, usage, articleType]
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Eğitim ve doğrulama setine ayırma
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[articleType], random_state=42)

# Dataset Sınıfı
class FashionDataset(Dataset)
    def __init__(self, dataframe, images_dir, transform=None)
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self)
        return len(self.dataframe)

    def __getitem__(self, idx)
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.images_dir, row[image_path])
        image = Image.open(img_path).convert(RGB)
        labels = torch.tensor(row[[season, year, usage, articleType]].values, dtype=torch.long)

        if self.transform
            image = self.transform(image)

        return image, labels

# Görüntü Dönüşümleri
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
train_dataset = FashionDataset(train_df, IMAGES_DIR, transform=transform_train)
val_dataset = FashionDataset(val_df, IMAGES_DIR, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model Sınıfı
class FashionModel(nn.Module)
    def __init__(self, num_classes_season, num_classes_year, num_classes_usage, num_classes_article)
        super(FashionModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.base_model = nn.Sequential(list(self.base_model.children())[-1])  # Fully connected katmanı çıkar
        self.flatten = nn.Flatten()

        # Çıkış katmanları
        self.fc_season = nn.Linear(2048, num_classes_season)
        self.fc_year = nn.Linear(2048, num_classes_year)
        self.fc_usage = nn.Linear(2048, num_classes_usage)
        self.fc_article = nn.Linear(2048, num_classes_article)

    def forward(self, x)
        x = self.base_model(x)
        x = self.flatten(x)
        season = self.fc_season(x)
        year = self.fc_year(x)
        usage = self.fc_usage(x)
        article = self.fc_article(x)
        return season, year, usage, article

# Modeli başlat
num_classes_season = len(label_encoders[season].classes_)
num_classes_year = len(label_encoders[year].classes_)
num_classes_usage = len(label_encoders[usage].classes_)
num_classes_article = len(label_encoders[articleType].classes_)

model = FashionModel(num_classes_season, num_classes_year, num_classes_usage, num_classes_article).cuda()

# Kayıp fonksiyonları ve optimizer
criterion_season = nn.CrossEntropyLoss()
criterion_year = nn.CrossEntropyLoss()
criterion_usage = nn.CrossEntropyLoss()
criterion_article = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Model Eğitim Fonksiyonu
def train_model_with_early_stopping(model, train_loader, val_loader, num_epochs=20, patience=5)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(num_epochs)
        model.train()
        running_loss, corrects = 0.0, [0]  4
        total = 0

        for inputs, labels in tqdm(train_loader, desc=fEpoch {epoch+1}{num_epochs} - Training)
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            losses = [
                criterion_season(outputs[0], labels[, 0]),
                criterion_year(outputs[1], labels[, 1]),
                criterion_usage(outputs[2], labels[, 2]),
                criterion_article(outputs[3], labels[, 3]),
            ]
            total_loss = sum(losses)

            # Backward
            total_loss.backward()
            optimizer.step()

            # Accuracy hesaplama
            for i in range(4)
                corrects[i] += (torch.max(outputs[i], 1)[1] == labels[, i]).sum().item()
            running_loss += total_loss.item()  inputs.size(0)
            total += inputs.size(0)

        print(fTrain Loss {running_loss  total.4f})
        for i, name in enumerate([season, year, usage, articleType])
            print(fTrain Accuracy ({name}) {corrects[i]  total.4f})

        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, [0]  4
        with torch.no_grad()
            for inputs, labels in tqdm(val_loader, desc=Validation)
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                losses = [
                    criterion_season(outputs[0], labels[, 0]),
                    criterion_year(outputs[1], labels[, 1]),
                    criterion_usage(outputs[2], labels[, 2]),
                    criterion_article(outputs[3], labels[, 3]),
                ]
                val_loss += sum(losses).item()  inputs.size(0)
                for i in range(4)
                    val_corrects[i] += (torch.max(outputs[i], 1)[1] == labels[, i]).sum().item()

        val_loss = len(val_loader.dataset)
        print(fValidation Loss {val_loss.4f})
        for i, name in enumerate([season, year, usage, articleType])
            print(fValidation Accuracy ({name}) {val_corrects[i]  len(val_loader.dataset).4f})

        # Early Stopping
        if val_loss  best_val_loss
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter = patience
                print(Early stopping triggered.)
                break

    # En iyi modeli yükle
    if best_model_weights
        model.load_state_dict(best_model_weights)

# Modeli eğit
train_model_with_early_stopping(model, train_loader, val_loader, num_epochs=20, patience=5)

# Son modeli kaydet
torch.save(model.state_dict(), fashion_model_multilabel.pth)
