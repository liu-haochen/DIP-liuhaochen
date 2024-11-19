import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from facades_dataset import FacadesDataset
from models import Generator, Discriminator  
import cv2
import numpy as np

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file=r'Assignments\03_PlayWithGANs\Pix2Pix\train_list.txt')
    val_dataset = FacadesDataset(list_file=r'Assignments\03_PlayWithGANs\Pix2Pix\val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=4)

    # Initialize models, loss functions, and optimizers
    generator = Generator().to(device)  # 生成器
    discriminator = Discriminator().to(device)  # 判别器

    criterion_GAN = nn.BCEWithLogitsLoss()  # 判别器损失函数
    criterion_L1 = nn.L1Loss()  # 用于计算生成图像与真实图像之间的L1损失
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))  # 生成器优化器
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))  # 判别器优化器

    # Add a learning rate scheduler for decay
    scheduler_G = StepLR(optimizer_G, step_size=200, gamma=0.2)
    scheduler_D = StepLR(optimizer_D, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        for i, (real_images, target_images) in enumerate(train_loader):
            real_images = real_images.to(device)
            target_images = target_images.to(device)
            

            # -----------------
            # 训练判别器
            # -----------------
            optimizer_D.zero_grad()

            # 判别器对真实图像的预测
            real_labels = torch.ones(real_images.size(0), 1).to(device)  # 真实图像标签为1
            fake_labels = torch.zeros(real_images.size(0), 1).to(device)  # 假图像标签为0
            real_labels_expanded = real_labels.view(-1, 1, 1, 1).expand(-1, -1, 15, 15)
            fake_labels_expanded = fake_labels.view(-1, 1, 1, 1).expand(-1, -1, 15, 15)
            # 判别器对真实图像的损失
            real_preds = discriminator(real_images, target_images)
            real_loss = criterion_GAN(real_preds, real_labels_expanded)

            # 判别器对生成图像的损失
            fake_images = generator(real_images)  # 生成图像
            fake_preds = discriminator(real_images, fake_images.detach())  # 不需要计算生成器的梯度
            fake_loss = criterion_GAN(fake_preds, fake_labels_expanded)

            # 计算判别器的总损失并反向传播
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # 训练生成器
            # -----------------
            optimizer_G.zero_grad()

            # 生成器损失：我们希望生成图像通过判别器，得到标签为“真”
            fake_preds = discriminator(real_images, fake_images)
            g_loss_GAN = criterion_GAN(fake_preds, real_labels_expanded)

            # 生成图像和目标图像的L1损失
            g_loss_L1 = criterion_L1(fake_images, target_images)

            # 生成器的总损失
            g_loss = g_loss_GAN + 100 * g_loss_L1  # L1损失权重为100
            g_loss.backward()
            optimizer_G.step()
            if epoch % 1 == 0 and i == 0:
                save_images(real_images, target_images, fake_images, r'Assignments\03_PlayWithGANs\Pix2Pix\train_results', epoch)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

        # 验证模型
        validate(generator, val_loader, criterion_L1, device, epoch, num_epochs)
        

        # Step the scheduler after each epoch
        scheduler_G.step()
        scheduler_D.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs(r'Assignments\03_PlayWithGANs\Pix2Pix\checkpoints', exist_ok=True)
            torch.save(generator.state_dict(), 
                       fr'Assignments\03_PlayWithGANs\Pix2Pix\checkpoints\pix2pix_generator_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), 
                       fr'Assignments\03_PlayWithGANs\Pix2Pix\checkpoints\pix2pix_discriminator_epoch_{epoch + 1}.pth')

def validate(generator, val_loader, criterion_L1, device, epoch, num_epochs):
    """
    Validation function for checking the performance of the model on the validation set.
    """
    generator.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i, (real_images, target_images) in enumerate(val_loader):
            real_images = real_images.to(device)
            target_images = target_images.to(device)

            fake_images = generator(real_images)
            val_loss += criterion_L1(fake_images, target_images).item()
            if epoch % 1 == 0 and i == 0:
                save_images(real_images, target_images, fake_images, r'Assignments\03_PlayWithGANs\Pix2Pix\val_results', epoch)
        avg_val_loss = val_loss / len(val_loader)
            
        print(f"Validation Loss at Epoch {epoch+1}/{num_epochs}: {avg_val_loss:.4f}")

    generator.train()
    

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)
def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

if __name__ == "__main__":
    main()