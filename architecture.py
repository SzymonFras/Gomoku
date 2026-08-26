import torch
model_1_structure = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, kernel_size=3, padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(128, 64, kernel_size=3, padding=1), torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 15 * 15, 225)
)
model_2_structure = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, kernel_size=5, padding=2), torch.nn.ReLU(),
    torch.nn.Conv2d(64, 128, kernel_size=5, padding=2), torch.nn.ReLU(),
    torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU(),
    torch.nn.Conv2d(128, 64, kernel_size=3, padding=1), torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 15 * 15, 225)
)
model_3_structure = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, kernel_size=5, padding=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(64, 128, kernel_size=5, padding=2), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.LeakyReLU(),
    torch.nn.Conv2d(128, 64, kernel_size=3, padding=1), torch.nn.LeakyReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 15 * 15, 225)
)
model_4_structure = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, kernel_size=5, padding=2), torch.nn.GELU(),
    torch.nn.Conv2d(64, 128, kernel_size=5, padding=2), torch.nn.GELU(),
    torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.GELU(),
    torch.nn.Conv2d(128, 64, kernel_size=3, padding=1), torch.nn.GELU(),
    torch.nn.Flatten(),
    torch.nn.Linear(64 * 15 * 15, 225)
)