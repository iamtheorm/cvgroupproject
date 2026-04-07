import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import numpy as np
import cv2

class DenseOpticalFlowRAFT:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=self.weights, progress=False).to(self.device)
        self.model.eval()
        self.transforms = self.weights.transforms()

    def compute_flow(self, img1, img2):
        '''
        Computes dense optical flow between img1 and img2.
        Images should be numpy arrays (H, W) or (H, W, C).
        '''
        # Convert grayscale to RGB if necessary as RAFT expects RGB
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        # Convert to torch tensor, shape [C, H, W]
        t1 = F.to_tensor(img1).unsqueeze(0).to(self.device)
        t2 = F.to_tensor(img2).unsqueeze(0).to(self.device)

        # Transform inputs
        t1, t2 = self.transforms(t1, t2)

        with torch.no_grad():
            list_of_flows = self.model(t1, t2)
            # The model outputs a list of predictions. The final one is the most accurate.
            predicted_flow = list_of_flows[-1][0]
        
        # Convert (2, H, W) to (H, W, 2) numpy array
        flow_np = predicted_flow.permute(1, 2, 0).cpu().numpy()
        return flow_np

    def process_sequence(self, sequence):
        '''
        Computes flow fields for the entire sequence.
        Returns a list of dense flow fields (length N-1).
        '''
        flows = []
        for i in range(len(sequence) - 1):
            flow = self.compute_flow(sequence[i], sequence[i+1])
            flows.append(flow)
        return flows
