import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from process_data import *

# Load the dataset using the custom class defined above
dataset = fake_and_real(root_path='data_train_antiface.csv', image_size=(112, 112))
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

# Load a pre-trained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)
# Freeze all layers to avoid updating them during training
for param in model.parameters():
    param.requires_grad = False
# Replace the final layer with a new linear layer to match the number of classes in the dataset
model.classifier[1] = nn.Linear(model.last_channel, 2)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the model
model.train()
for epoch in range(0,NUM_EPOCHS+1):
    running_loss = 0.0
    
    for inputs, labels in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        labels = labels.unsqueeze(1).expand(-1, outputs.size(1))
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'\nEpoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader)}\n')
    # save model
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if epoch % SAVED_EPOCH == 0:
        torch.save(model.state_dict(),
            os.path.join(SAVE_DIR, '%03d.pth' % epoch))
# Save the trained model
torch.save(model.state_dict(), 'real_and_fake_face_model.pth')
