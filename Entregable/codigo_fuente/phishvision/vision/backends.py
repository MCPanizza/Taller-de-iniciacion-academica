import torch
from PIL import Image
import timm
from typing import List
import numpy as np
from torchvision import transforms
from tqdm import tqdm

class VisionEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.transform = None
        
        self._load_model()
        
    def _load_model(self):
        print(f"Cargando modelo: {self.model_name} en {self.device}...")
        
        if "clip" in self.model_name.lower():
            # OpenAI CLIP via TIMM (ViT-L-14)
            model_id = "vit_large_patch14_clip_224.openai"
        elif "siglip" in self.model_name.lower():
            # SigLIP via TIMM
            model_id = "vit_base_patch16_siglip_224"
        elif "dinov2" in self.model_name.lower():
            # DINOv2 via TIMM
            model_id = "vit_base_patch14_dinov2.lvd142m"
        else:
            raise ValueError(f"Modelo no soportado: {self.model_name}")
            
        # Cargar modelo TIMM
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).to(self.device)
        self.model.eval()
        
        # Configurar transformaciones
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
            
    def encode_image(self, image_paths: List[str]) -> np.ndarray:
        images = []
        print(f"Cargando {len(image_paths)} imágenes...")
        
        for path in tqdm(image_paths, desc="Loading Images"):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error cargando imagen {path}: {e}")
                # Placeholder negro en caso de error
                images.append(Image.new("RGB", (224, 224)))
                
        if not images:
            return np.array([])
            
        embeddings = []
        
        print(f"Generando embeddings con {self.model_name}...")
        with torch.no_grad():
            batch_size = 16
            for i in tqdm(range(0, len(images), batch_size), desc="Encoding Batches"):
                batch = images[i:i+batch_size]
                # Aplicar transformaciones
                tensors = torch.stack([self.transform(img) for img in batch]).to(self.device)
                
                # Inferencia
                outputs = self.model(tensors)
                
                # Normalizar (importante para similitud coseno)
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(outputs.cpu().numpy())
        
        return np.concatenate(embeddings)
