import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
class FeatureEnhancer(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(FeatureEnhancer, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.norm2 = nn.LayerNorm(in_dim)
        self.act2 = nn.ELU(inplace=True)
        
    def forward(self, x):
        identity = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = x + identity
        x = self.act2(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0)
    
    def forward(self, query, key, value):
        residual = query
        query = self.norm1(query)
        
        batch_size = query.size(0)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(batch_size, -1, self.num_heads, q.size(-1) // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, k.size(-1) // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, v.size(-1) // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, q.size(-1) * self.num_heads)
        output = self.out_proj(output)
        output = output + residual
        output = self.norm2(output)
        
        return output

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class StageTwo(nn.Module):
    def __init__(self, num_tools, num_actions, num_targets, num_triplets, base_model='resnet18'):
        super(StageTwo, self).__init__()
        self.base_model = base_model
        if base_model == 'resnet18':
            self.feature_dim = 512
            self.hidden_dim = 1024
        elif base_model == 'resnet34':
            self.feature_dim = 512
            self.hidden_dim = 1024
        elif base_model == 'resnet50':
            self.feature_dim = 2048
            self.hidden_dim = 4096
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        self.num_tools = num_tools
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.num_triplets = num_triplets
        self.tool_encoder = self._build_encoder(base_model)
        self.orig_encoder = self._build_encoder(base_model)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.tool_enhancer = nn.Sequential(
            nn.Flatten(),
            FeatureEnhancer(self.feature_dim, self.hidden_dim)
        )
        self.orig_enhancer = nn.Sequential(
            nn.Flatten(),
            FeatureEnhancer(self.feature_dim, self.hidden_dim)
        )
        self.cross_attn_action = CrossAttention(self.feature_dim)
        self.cross_attn_target = CrossAttention(self.feature_dim)
        self.action_head = ClassificationHead(
            self.feature_dim, 
            self.hidden_dim, 
            num_actions
        )
        self.target_head = ClassificationHead(
            self.feature_dim, 
            self.hidden_dim, 
            num_targets
        )
        self.triplet_decoder = TripletDecoder(num_tools, num_actions, num_targets, num_triplets)
    
    def _build_encoder(self, base_model):
        if base_model == 'resnet18':
            encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif base_model == 'resnet50':
            encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        return nn.Sequential(*list(encoder.children())[:-2])
    
    def forward(self, inputs):
        crop_img = inputs['crop_img']
        original_img = inputs['original_img']
        tool_feat_maps = self.tool_encoder(crop_img)
        tool_pooled = self.gap(tool_feat_maps)
        tool_features = self.tool_enhancer(tool_pooled)
        orig_feat_maps = self.orig_encoder(original_img)
        orig_pooled = self.gap(orig_feat_maps)
        orig_features = self.orig_enhancer(orig_pooled)
        B, C, H, W = tool_feat_maps.size()
        tool_feats_flat = tool_feat_maps.view(B, C, -1).permute(0, 2, 1)
        
        orig_feats_flat = orig_feat_maps.view(B, C, -1).permute(0, 2, 1)
        action_context = self.cross_attn_action(
            query=orig_feats_flat,
            key=tool_feats_flat,
            value=tool_feats_flat
        )
        
        target_context = self.cross_attn_target(
            query=orig_feats_flat,
            key=tool_feats_flat,
            value=tool_feats_flat
        )
        action_feats = action_context.mean(dim=1)
        target_feats = target_context.mean(dim=1)
        action_logits = self.action_head(action_feats)
        target_logits = self.target_head(target_feats)
        if isinstance(inputs, dict) and 'tool_probs' in inputs:
            tool_logits = inputs['tool_probs']
        else:
            assert False
        triplet_logits = self.triplet_decoder(tool_logits, action_logits, target_logits)
        return {
            'tool_logits': tool_logits,
            'action_logits': action_logits,
            'target_logits': target_logits,
            'triplet_logits': triplet_logits,
            'features': tool_features
        }


class TripletDecoder(nn.Module):
    def __init__(self, num_tools, num_actions, num_targets, num_triplets):
        super(TripletDecoder, self).__init__()
        self.tool_transform = nn.Parameter(torch.randn(num_tools, num_tools))
        self.action_transform = nn.Parameter(torch.randn(num_actions, num_actions))
        self.target_transform = nn.Parameter(torch.randn(num_targets, num_targets))
        self.bn_tool = nn.BatchNorm1d(num_tools)
        self.bn_action = nn.BatchNorm1d(num_actions)
        self.bn_target = nn.BatchNorm1d(num_targets)
        self.mlp = nn.Sequential(
            nn.Linear(num_triplets, num_triplets*2),
            nn.BatchNorm1d(num_triplets*2),
            nn.ELU(),
            nn.Linear(num_triplets*2, num_triplets)
        )
        self.elu = nn.ELU(inplace=True)
        self.num_tools = num_tools
        self.num_actions = num_actions
        self.num_targets = num_targets
        self.num_triplets = num_triplets
        self.valid_position = self._load_valid_triplets("/ssd/prostate/dataset_triplet/triplet_maps.txt")
        if self.valid_position is not None and self.valid_position.shape[0] != self.num_triplets:
            print(f"Warning: Valid triplet count ({self.valid_position.shape[0]}) does not match num_triplets ({self.num_triplets})")
    
    def _load_valid_triplets(self, map_file):
        valid_indices = []
        
        try:
            with open(map_file, 'r') as f:
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        _, tool_id, action_id, target_id = map(int, parts[:4])
                        if (tool_id < self.num_tools and action_id < self.num_actions 
                            and target_id < self.num_targets):
                            index = tool_id * (self.num_actions * self.num_targets) + action_id * self.num_targets + target_id
                            valid_indices.append(index)
            if valid_indices:
                return torch.tensor(valid_indices)
            return None
        except Exception as e:
            print(f"Failed to load mask file: {e}")
            return None
    
    def mask(self, ivts):
        if self.valid_position is not None:
            ivt_flatten = ivts.reshape([-1, self.num_tools * self.num_actions * self.num_targets])
            n = ivt_flatten.shape[0]
            valid_position = torch.stack([self.valid_position] * n, dim=0).to(ivts.device)
            valid_triplets = torch.gather(input=ivt_flatten, dim=-1, index=valid_position)
            return valid_triplets
        
        return ivts
    
    def forward(self, tool_logits, action_logits, target_logits):
        tool = torch.matmul(tool_logits, self.tool_transform)
        tool = self.elu(self.bn_tool(tool))
        
        action = torch.matmul(action_logits, self.action_transform)
        action = self.elu(self.bn_action(action))
        
        target = torch.matmul(target_logits, self.target_transform)
        target = self.elu(self.bn_target(target))
        ivt_maps = torch.einsum('bi,bj,bk->bijk', tool, action, target)
        ivt_masked = self.mask(ivts=ivt_maps)
        triplet_logits = self.mlp(ivt_masked)
        
        return triplet_logits
StageTwoWithPaired = StageTwo
