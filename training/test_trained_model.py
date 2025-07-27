import torch
import pickle
import os
from models.dilated_tooth_seg_network import DilatedToothSegmentationNetwork

def test_trained_model():
    """Test the trained model on a single example and compare with untrained model."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This model requires GPU for PointNet2 operations.")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    
    # Check if trained model exists
    model_path = "small_trained_model.pth"
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}")
        return
    
    # Load test data
    processed_dir = "data/3dteethseg/processed"
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    # Use a different file than the ones used in training
    test_file = processed_files[5] if len(processed_files) > 5 else processed_files[0]
    file_path = os.path.join(processed_dir, test_file)
    
    print(f"Testing on: {test_file}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    pos, x, labels = data
    
    # Create untrained model
    num_classes = 17
    feature_dim = 24
    untrained_model = DilatedToothSegmentationNetwork(num_classes=num_classes, feature_dim=feature_dim).to(device)
    
    # Load trained model
    trained_model = DilatedToothSegmentationNetwork(num_classes=num_classes, feature_dim=feature_dim).to(device)
    trained_model.load_state_dict(torch.load(model_path))
    
    # Prepare input data
    pos_batch = pos.unsqueeze(0).to(device)
    x_batch = x.unsqueeze(0).to(device)
    
    print(f"Testing on {pos.shape[0]} points...")
    
    # Test untrained model
    untrained_model.eval()
    with torch.no_grad():
        untrained_output = untrained_model(x_batch, pos_batch)
        untrained_predictions = torch.argmax(untrained_output, dim=2)
        untrained_accuracy = (untrained_predictions[0] == labels.to(device)).float().mean()
    
    # Test trained model
    trained_model.eval()
    with torch.no_grad():
        trained_output = trained_model(x_batch, pos_batch)
        trained_predictions = torch.argmax(trained_output, dim=2)
        trained_accuracy = (trained_predictions[0] == labels.to(device)).float().mean()
    
    print(f"\nResults:")
    print(f"  Untrained model accuracy: {untrained_accuracy.item():.4f}")
    print(f"  Trained model accuracy: {trained_accuracy.item():.4f}")
    print(f"  Improvement: {trained_accuracy.item() - untrained_accuracy.item():.4f}")
    
    # Show some prediction details
    print(f"\nPrediction details:")
    print(f"  Ground truth classes: {torch.unique(labels)}")
    print(f"  Untrained predictions: {torch.unique(untrained_predictions[0])}")
    print(f"  Trained predictions: {torch.unique(trained_predictions[0])}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_trained_model() 