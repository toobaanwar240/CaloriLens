"""
Smart Ensemble Food Classifier - FIXED VERSION
Combines Model 1 (11 classes) + Model 2 (49 classes) = 54 total classes
Intelligently routes predictions to the right model
"""

import torch
import timm
import numpy as np
from typing import Dict, List
from PIL import Image
from torchvision import transforms

from .class_labels import (
    MODEL1_LABELS, MODEL2_LABELS, 
    UNIQUE_TO_MODEL1, UNIQUE_TO_MODEL2, 
    OVERLAPPING_LABELS, ALL_LABELS
)


class SmartEnsembleClassifier:
    """
    Intelligent ensemble that knows which model to trust for each class
    """
    
    def __init__(
        self,
        model1_weights: str,
        model2_weights: str,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            model1_weights: Path to food_classifier_finetuned (1).pth
            model2_weights: Path to food_classifier_finetuned (3).pth
            confidence_threshold: Minimum confidence to consider prediction
        """
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        
        print(f"🚀 Initializing Smart Ensemble Classifier")
        print(f"🖥️  Device: {self.device}")
        
        # Load Model 1 (11 classes - has pizza, ramen, etc.)
        print(f"\n📦 Loading Model 1 (11 classes)...")
        print(f"   Unique classes: {UNIQUE_TO_MODEL1}")
        self.model1 = timm.create_model(
            'convnext_tiny',
            pretrained=False,
            num_classes=11
        ).to(self.device)
        
        try:
            state_dict1 = torch.load(model1_weights, map_location=self.device)
            if 'model_state_dict' in state_dict1:
                self.model1.load_state_dict(state_dict1['model_state_dict'])
            else:
                self.model1.load_state_dict(state_dict1)
            self.model1.eval()
            print("   ✅ Model 1 loaded")
        except Exception as e:
            print(f"   ❌ Error loading Model 1: {e}")
            self.model1 = None
        
        # Load Model 2 (49 classes - Food-101)
        print(f"\n📦 Loading Model 2 (49 classes)...")
        print(f"   Contains {len(UNIQUE_TO_MODEL2)} unique classes")
        self.model2 = timm.create_model(
            'convnext_tiny',
            pretrained=False,
            num_classes=49
        ).to(self.device)
        
        try:
            state_dict2 = torch.load(model2_weights, map_location=self.device)
            if 'model_state_dict' in state_dict2:
                self.model2.load_state_dict(state_dict2['model_state_dict'])
            else:
                self.model2.load_state_dict(state_dict2)
            self.model2.eval()
            print("   ✅ Model 2 loaded")
        except Exception as e:
            print(f"   ❌ Error loading Model 2: {e}")
            self.model2 = None
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"\n✅ Ensemble ready!")
        print(f"   Total coverage: {len(ALL_LABELS)} unique food classes")
        print(f"   Pizza, ramen, sushi, steak, samosa: ✅ Supported (Model 1)")
    
    def predict(self, image: np.ndarray, top_k: int = 3) -> Dict:
        """
        Smart prediction using both models
        
        Strategy:
        1. Run both models
        2. If Model 1 detects unique classes (pizza, ramen, etc.) with high confidence, use it
        3. If Model 2 detects unique classes with high confidence, use it
        4. For overlapping classes, use model with higher confidence
        5. Return top-k predictions from the chosen model
        """
        
        # Initialize return dict with defaults
        result = {
            'label': 'unknown',
            'confidence': 0.0,
            'top_k': [],
            'chosen_model': None,
            'reasoning': 'No prediction made',
            'all_predictions': {}
        }
        
        try:
            # Preprocess image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image[:, :, ::-1]  # BGR to RGB
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Get predictions from both models
            predictions = {}
            
            with torch.no_grad():
                # Model 1 predictions
                if self.model1 is not None:
                    try:
                        logits1 = self.model1(img_tensor)
                        probs1 = torch.softmax(logits1, dim=1)[0]
                        top_probs1, top_indices1 = torch.topk(probs1, min(top_k, 11))
                        
                        model1_results = []
                        for prob, idx in zip(top_probs1, top_indices1):
                            model1_results.append({
                                'label': MODEL1_LABELS[idx.item()],
                                'confidence': prob.item()
                            })
                        predictions['model1'] = model1_results
                    except Exception as e:
                        print(f"⚠️ Model 1 prediction failed: {e}")
                
                # Model 2 predictions
                if self.model2 is not None:
                    try:
                        logits2 = self.model2(img_tensor)
                        probs2 = torch.softmax(logits2, dim=1)[0]
                        top_probs2, top_indices2 = torch.topk(probs2, min(top_k, 49))
                        
                        model2_results = []
                        for prob, idx in zip(top_probs2, top_indices2):
                            model2_results.append({
                                'label': MODEL2_LABELS[idx.item()],
                                'confidence': prob.item()
                            })
                        predictions['model2'] = model2_results
                    except Exception as e:
                        print(f"⚠️ Model 2 prediction failed: {e}")
            
            # Update result with all predictions
            result['all_predictions'] = predictions
            
            # Smart decision logic - FINAL BALANCED APPROACH
            if 'model1' in predictions and 'model2' in predictions:
                pred1_top = predictions['model1'][0]
                pred2_top = predictions['model2'][0]
                
                conf1 = pred1_top['confidence']
                conf2 = pred2_top['confidence']
                label1 = pred1_top['label']
                label2 = pred2_top['label']
                
                # CRITICAL RULE 1: If BOTH models are very confident about their unique classes,
                # trust the MORE confident one (this handles pizza vs chicken_quesadilla)
                if (label1 in UNIQUE_TO_MODEL1 and conf1 > 0.95 and 
                    label2 in UNIQUE_TO_MODEL2 and conf2 > 0.95):
                    if conf1 > conf2:
                        result['chosen_model'] = 'model1'
                        result['top_k'] = predictions['model1']
                        result['reasoning'] = f"Both very confident, Model 1 higher: '{label1}' ({conf1:.2%} vs {conf2:.2%})"
                    else:
                        result['chosen_model'] = 'model2'
                        result['top_k'] = predictions['model2']
                        result['reasoning'] = f"Both very confident, Model 2 higher: '{label2}' ({conf2:.2%} vs {conf1:.2%})"
                
                # CRITICAL RULE 2: Model 2 more confident about its unique class than Model 1 overall
                elif label2 in UNIQUE_TO_MODEL2 and conf2 > conf1 and conf2 > 0.95:
                    result['chosen_model'] = 'model2'
                    result['top_k'] = predictions['model2']
                    result['reasoning'] = f"Model 2 MORE confident about '{label2}' ({conf2:.2%} vs {conf1:.2%})"
                
                # PRIORITY 1: Model 1 unique class with very high confidence
                elif label1 in UNIQUE_TO_MODEL1 and conf1 > 0.95:
                    result['chosen_model'] = 'model1'
                    result['top_k'] = predictions['model1']
                    result['reasoning'] = f"Model 1 very confident about '{label1}' ({conf1:.2%})"
                
                # PRIORITY 2: Model 2 unique class with very high confidence
                elif label2 in UNIQUE_TO_MODEL2 and conf2 > 0.95:
                    result['chosen_model'] = 'model2'
                    result['top_k'] = predictions['model2']
                    result['reasoning'] = f"Model 2 very confident about '{label2}' ({conf2:.2%})"
                
                # PRIORITY 3: Model 1 unique class with good confidence
                elif label1 in UNIQUE_TO_MODEL1 and conf1 > 0.85 and conf1 > conf2:
                    result['chosen_model'] = 'model1'
                    result['top_k'] = predictions['model1']
                    result['reasoning'] = f"Model 1 confident about '{label1}' ({conf1:.2%})"
                
                # PRIORITY 4: Model 2 unique class with good confidence
                elif label2 in UNIQUE_TO_MODEL2 and conf2 > 0.85 and conf2 > conf1:
                    result['chosen_model'] = 'model2'
                    result['top_k'] = predictions['model2']
                    result['reasoning'] = f"Model 2 confident about '{label2}' ({conf2:.2%})"
                
                # PRIORITY 5: Both predict overlapping class - use higher confidence
                elif label1 in OVERLAPPING_LABELS and label2 in OVERLAPPING_LABELS:
                    if conf1 > conf2:
                        result['chosen_model'] = 'model1'
                        result['top_k'] = predictions['model1']
                        result['reasoning'] = f"Both know '{label1}', Model 1 more confident ({conf1:.2%})"
                    else:
                        result['chosen_model'] = 'model2'
                        result['top_k'] = predictions['model2']
                        result['reasoning'] = f"Both know '{label2}', Model 2 more confident ({conf2:.2%})"
                
                # PRIORITY 6: Significant confidence difference
                elif conf1 > conf2 + 0.15:
                    result['chosen_model'] = 'model1'
                    result['top_k'] = predictions['model1']
                    result['reasoning'] = f"Model 1 much more confident ({conf1:.2%} vs {conf2:.2%})"
                
                elif conf2 > conf1 + 0.15:
                    result['chosen_model'] = 'model2'
                    result['top_k'] = predictions['model2']
                    result['reasoning'] = f"Model 2 much more confident ({conf2:.2%} vs {conf1:.2%})"
                
                # DEFAULT: Use higher confidence
                else:
                    if conf1 > conf2:
                        result['chosen_model'] = 'model1'
                        result['top_k'] = predictions['model1']
                        result['reasoning'] = f"Model 1 more confident ({conf1:.2%} vs {conf2:.2%})"
                    else:
                        result['chosen_model'] = 'model2'
                        result['top_k'] = predictions['model2']
                        result['reasoning'] = f"Model 2 more confident ({conf2:.2%} vs {conf1:.2%})"
            
            elif 'model1' in predictions:
                result['chosen_model'] = 'model1'
                result['top_k'] = predictions['model1']
                result['reasoning'] = "Only Model 1 available"
            
            elif 'model2' in predictions:
                result['chosen_model'] = 'model2'
                result['top_k'] = predictions['model2']
                result['reasoning'] = "Only Model 2 available"
            
            else:
                result['error'] = 'No models available or all predictions failed'
                return result
            
            # Update label and confidence from top prediction
            if result['top_k']:
                result['label'] = result['top_k'][0]['label']
                result['confidence'] = result['top_k'][0]['confidence']
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        
        return result
    
    def predict_batch(self, images: List[np.ndarray], top_k: int = 3) -> List[Dict]:
        """Predict multiple images"""
        results = []
        for image in images:
            result = self.predict(image, top_k)
            results.append(result)
        return results
    
    def get_supported_classes(self) -> List[str]:
        """Return all supported food classes"""
        return ALL_LABELS.copy()
    
    def get_model_info(self) -> Dict:
        """Get information about the ensemble"""
        return {
            'total_classes': len(ALL_LABELS),
            'model1_classes': MODEL1_LABELS,
            'model2_classes': MODEL2_LABELS,
            'unique_to_model1': UNIQUE_TO_MODEL1,
            'unique_to_model2': len(UNIQUE_TO_MODEL2),
            'overlapping': OVERLAPPING_LABELS,
            'device': self.device
        }