import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCrossAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2, proj_channels):
        super().__init__()
        self.proj_channels = proj_channels
        self.query_conv = nn.Conv2d(in_channels1, proj_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels2, proj_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels2, in_channels2, kernel_size=1)
        
        self.out_conv = nn.Conv2d(in_channels2, in_channels2, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, f1, f2):
        if f1.shape[2:] != f2.shape[2:]:
            f1 = F.interpolate(f1, size=f2.shape[2:], mode='bilinear', align_corners=False)
        batch_size, _, height, width = f2.size()
        # Generate Q, K, V
        query = self.query_conv(f1).view(batch_size, self.proj_channels, -1) # B, C, HW
        key = self.key_conv(f2).view(batch_size, self.proj_channels, -1).permute(0, 2, 1) # B, HW, C
        value = self.value_conv(f2).view(batch_size, -1, height * width) # B, C_in, HW

        key = F.softmax(key, dim=-1)  # softmax over the key channel dimension
        context = torch.bmm(value, key) # (B, C_in, HW) x (B, HW, C) -> (B, C_in, C)
        
        # 2. Dot-product between context and query
        attended_value = torch.bmm(context, query) # (B, C_in, C) x (B, C, HW) -> (B, C_in, HW)
        attended_value = attended_value.view(batch_size, -1, height, width)
        # Residual connection
        output = f2 + self.gamma * self.out_conv(attended_value)
        
        return output